"""Public HTTP checks on an ALREADY OWNED pair. Import only; no standalone launch or engine shutdown.

The caller owns engine/peer/samplers and must close them in its existing finally block.
Uses the product Service, Server and handlers on a loopback ephemeral port. Optional
hardware/queue-watch samplers are omitted; the outer validation driver collects them.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import http.client
import json
from pathlib import Path
import socket
import threading
import time


def run_http_smoke(*, engine, pack, tok, runtime, family, folder, cfg, render=None):
    """Return JSON evidence; start/reap only a local HTTP listener. Never launch/restart/stop the pair.

    Required: idle owned rank0 engine, READY multi=4, GLM32K or DeepSeek8K,
    matching runtime package and frozen engine on both ranks. cfg is read-only.
    render is accepted for the driver's common hook but Service selects the real family template.
    """
    from serve.frontend import ChatTemplate
    from serve.server import Service, Server, make_handler, chat_template_path

    assert family in ("glm", "ds4"), family
    assert engine.multi == 4 and not engine._routes, engine.info
    assert engine.info.get("tp_world") == 2 and engine.info.get("tp_pair") != "down", engine.info
    expected_context = 32768 if family == "glm" else 8192
    assert engine.max_context == expected_context, (engine.max_context, expected_context)
    pack, runtime, folder = Path(pack), Path(runtime), Path(folder)
    model = "maya-glm-s-http-validation" if family == "glm" else "maya-deepseek-xs-http-validation"
    template = ChatTemplate(chat_template_path(pack / "tokenizer"))
    assert template.deepseek == (family == "ds4"), (family, template.source[:80])
    svc = Service(engine, tok, template, model_name=model, sampling_defaults={})
    svc.fifo.set_capacity(4)  # same setup as server.main after READY multi=4
    svc.backend = "hip"
    svc.loop_guard = False  # bounded plain-answer requests, no automatic continuation/retry
    svc.dspark, svc.lookup = (True, False) if family == "ds4" else (None, None)
    svc.model_label = model
    off = 1 if family == "glm" else 0
    report = {"family": family, "context": expected_context, "slots": 4,
              "scope": "real HTTP handlers and Service on existing owned pair; no model load or engine shutdown",
              "runtime": str(runtime), "started": time.time(), "requests": [], "captures": []}
    evidence_path = folder / "http-smoke.json"
    lock = threading.Lock()
    active, peak, active_threads = set(), 0, set()
    saved_generate = engine.__dict__.get("generate")
    had_generate = "generate" in engine.__dict__
    original_generate = engine.generate

    def persist():
        with lock:
            evidence = copy.deepcopy(report)
        evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")

    def traced_generate(ids, max_new, sampling, cancel, embeddings=None):
        nonlocal peak
        label = (sampling or {}).get("user")
        assert isinstance(label, str) and label.startswith("http-"), sampling
        row = {"label": label, "ids": list(ids), "max_new": max_new,
               "sampling": copy.deepcopy(sampling), "tokens": [], "started": time.monotonic()}
        thread = threading.current_thread()
        with lock:
            active.add(label)
            active_threads.add(thread)
            peak = max(peak, len(active))
        gen = original_generate(ids, max_new, sampling, cancel, embeddings)
        try:
            for token in gen:
                if token is not None:
                    row["tokens"].append(int(token))
                yield token
        except BaseException as error:
            if not isinstance(error, GeneratorExit):
                row["error"] = repr(error)
            raise
        finally:
            # Original generator drains the request's DONE on close; read thread-local figures only afterwards.
            try:
                gen.close()
            finally:
                row.update(last=engine.last.copy(), finished=time.monotonic(), cancelled=cancel.is_set())
                with lock:
                    report["captures"].append(row)
                    active.discard(label)
                    active_threads.discard(thread)

    class OwnedServer(Server):
        # A failed GPU request must not block the caller's finally, which owns engine teardown.
        # Success below requires every request thread and route to finish before listener cleanup.
        daemon_threads = True

    engine.generate = traced_generate
    httpd = None
    listener = None
    success = False
    try:
        httpd = OwnedServer(("127.0.0.1", 0), make_handler(svc))
        httpd.response_service = svc
        port = httpd.server_address[1]
        svc.listen = ("127.0.0.1", port)
        report["listen"] = ["127.0.0.1", port]
        listener = threading.Thread(target=httpd.serve_forever, name="maya-owned-http-smoke", daemon=True)
        listener.start()

        def get(path):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
            try:
                conn.request("GET", path)
                response = conn.getresponse()
                assert response.status == 200, (path, response.status, response.read())
                return json.loads(response.read())
            finally:
                conn.close()

        def health():
            value = get("/health")
            assert value["status"] == "ok", value
            assert value["model"] == model and value["max_context"] == expected_context, value
            assert value["tp"]["world_size"] == 2 and value["tp"]["peer"] == "up", value
            return value

        def captured(label, timeout=30):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                with lock:
                    rows = [row for row in report["captures"] if row["label"] == label]
                if rows:
                    assert len(rows) == 1 and "error" not in rows[0], rows
                    return rows[0]
                time.sleep(.025)
            raise AssertionError("engine DONE capture missing: " + label)

        def request(label, prompt, *, stream=False, spec=True, budget=64, seed=1234, sampled=False,
                    cancel_chunks=None, barrier=None):
            body = {"model": model, "user": label, "messages": [{"role": "user", "content": prompt}],
                    "temperature": .7 if sampled else 0, "seed": seed, "max_tokens": budget,
                    "reasoning_effort": "none", "stream": stream}
            if sampled:
                body["top_p"] = .9
            if not spec:
                body["strata_tune"] = {"spec": off}
            if barrier:
                barrier.wait(timeout=10)
            row = {"label": label, "body": body, "started": time.monotonic(), "stream": stream}
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
            try:
                conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                             headers={"Content-Type": "application/json", "Connection": "close"})
                response = conn.getresponse()
                assert response.status == 200, (label, response.status, response.read().decode())
                row["status"] = response.status
                if stream:
                    assert response.getheader("Content-Type", "").startswith("text/event-stream")
                    chunks, text, done = [], [], False
                    deadline = time.monotonic() + 120
                    count = 0
                    while True:
                        assert time.monotonic() < deadline, label
                        line = response.readline()
                        if not line:
                            break
                        if not line.startswith(b"data:"):
                            continue  # heartbeat comments and blank SSE delimiters
                        payload = line[5:].strip()
                        if payload == b"[DONE]":
                            done = True
                            break
                        chunk = json.loads(payload)
                        assert "error" not in chunk, (label, chunk)
                        chunks.append(chunk)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        part = delta.get("content") or delta.get("reasoning_content")
                        if part:
                            text.append(part)
                            count += 1
                        if cancel_chunks and count >= cancel_chunks:
                            # Explicit TCP EOF/reset wakes watch_client even if a buffered response object survives.
                            sock = conn.sock or getattr(getattr(response.fp, "raw", None), "_sock", None)
                            if sock is not None:
                                try:
                                    sock.shutdown(socket.SHUT_RDWR)
                                except OSError:
                                    pass
                            response.close()
                            row["client_cancelled"] = True
                            break
                    row.update(chunks=chunks, text="".join(text), sse_done=done)
                    if not cancel_chunks:
                        assert done and chunks and chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length"), row
                        row["usage"] = chunks[-1]["usage"]
                else:
                    answer = json.loads(response.read())
                    assert "error" not in answer, answer
                    assert answer["choices"][0]["finish_reason"] in ("stop", "length"), answer
                    row.update(answer=answer, text=answer["choices"][0]["message"].get("content", ""),
                               usage=answer["usage"])
                if not cancel_chunks:
                    assert 0 < row["usage"]["completion_tokens"] <= budget, row
                    assert row["text"], row
            finally:
                conn.close()
                row["finished"] = time.monotonic()
                row["wall_s"] = row["finished"] - row["started"]
                with lock:
                    report["requests"].append(row)
            evidence = captured(label)
            assert evidence["tokens"] and len(evidence["tokens"]) <= budget, evidence
            if cancel_chunks:
                assert evidence["last"].get("finish") == "cancel" and row.get("client_cancelled"), (row, evidence)
            elif not spec:
                assert evidence["last"].get("drafts_offered", 0) == 0, evidence
            return row

        report["health_before"] = health()
        prompt = "Write an eBPF tutorial with verifier safety, maps and tracing examples. Explain each step clearly."
        plain = request("http-off", prompt, spec=False)
        streamed = request("http-on", prompt, stream=True)
        off_capture, on_capture = captured("http-off"), captured("http-on")
        assert off_capture["ids"] == on_capture["ids"] and off_capture["tokens"] == on_capture["tokens"], report
        assert plain["text"] == streamed["text"], (plain, streamed)
        assert on_capture["last"].get("drafts_offered", 0) > 0, on_capture
        report["spec_parity"] = {"input_ids_equal": True, "output_ids_equal": True,
                                 "http_text_equal": True, "off_value": off}
        wave_started = time.time()
        barrier = threading.Barrier(4)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(request, "http-wave-" + str(i),
                                   f"Explain eBPF tracing scenario {i}, including maps, safety and a concrete example.",
                                   stream=bool(i % 2), spec=i != 0, budget=64, seed=1234 + i,
                                   sampled=i == 1, barrier=barrier) for i in range(4)]
            wave = [future.result(timeout=150) for future in futures]
        assert peak == 4, {"peak_engine_request_calls": peak}
        stats = [dict(row) for row in engine.stat_history.copy() if row.get("time", 0) >= wave_started]
        engine_peak = max((row.get("active", 0) for row in stats), default=0)
        assert engine_peak >= 4, {"missing_engine_four_active": stats}
        for i in range(1, 4):
            assert captured("http-wave-" + str(i))["last"].get("drafts_offered", 0) > 0
        report["wave"] = {"peak_engine_request_calls": peak, "peak_stat_active": engine_peak,
                          "engine_stat_scope": "engine-wide samples during HTTP wave", "stat_samples": stats,
                          "wall_s": max(row["finished"] for row in wave) - min(row["started"] for row in wave),
                          "completion_tokens": sum(row["usage"]["completion_tokens"] for row in wave)}
        long_prompt = "Write a detailed eBPF coding tutorial with at least 20 numbered examples and explanations."
        with ThreadPoolExecutor(max_workers=2) as pool:
            cancelled = pool.submit(request, "http-cancel", long_prompt, stream=True, budget=256, cancel_chunks=3)
            survivor = pool.submit(request, "http-survivor", prompt + " Include a map example.", stream=True, budget=64)
            cancelled.result(timeout=150)
            with lock:
                survivor_active_at_refill = "http-survivor" in active
            followup = request("http-followup", "Explain why the eBPF verifier rejects unbounded loops and unsafe pointers.", budget=32)
            survivor.result(timeout=150)
        survivor_capture, refill_capture = captured("http-survivor"), captured("http-followup")
        report["cancellation_refill"] = {
            "survivor_active_at_refill": survivor_active_at_refill,
            "engine_calls_overlap": min(survivor_capture["finished"], refill_capture["finished"]) >
                                    max(survivor_capture["started"], refill_capture["started"]),
            "scope": "observed overlap; timing-dependent and not a requirement for success",
            "followup_status": followup["status"]}
        assert not engine._routes and not active, {"routes": list(engine._routes), "active": list(active)}
        report["health_after"] = health()
        report["history"] = list(svc.history)
        report["success"] = success = True
        return report
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        svc.stopping = True
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()  # closes response jobs, never the injected engine
        if listener is not None:
            listener.join(timeout=5)
        with lock:
            workers = list(active_threads)
        for worker in workers:
            worker.join(timeout=5)
        with lock:
            still_active = [thread.name for thread in active_threads if thread.is_alive()]
        report["cleanup"] = {"listener_exited": listener is None or not listener.is_alive(),
                             "active_http_workers": still_active, "active_engine_routes": list(engine._routes),
                             "engine_alive": engine.alive(), "pair_unchanged": engine.info.get("tp_pair") != "down"}
        # Do not detach the wrapper from an unfinished generator; the caller will close its owned pair on failure.
        if not still_active:
            if had_generate:
                engine.generate = saved_generate
            else:
                del engine.__dict__["generate"]
        report["finished"] = time.time()
        persist()
        if success:
            assert report["cleanup"]["listener_exited"] and not still_active and not engine._routes, report["cleanup"]
            assert report["cleanup"]["engine_alive"] and report["cleanup"]["pair_unchanged"], report["cleanup"]
