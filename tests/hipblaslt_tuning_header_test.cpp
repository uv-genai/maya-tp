// hipblaslt_tuning_header_test - CPU only. The tuning-table loader accepts upstream's MAYA_HIPBLASLT_TUNING_V1 header
// and the legacy STRATA_HIPBLASLT_TUNING_V1 one the fork's committed tables keep (engines built before upstream
// v1.1.59 read only the legacy name, and each TP2 rank reads its own host's tables), and still rejects anything else.
#include "../src/prefill/hipblaslt_tuning.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

#ifndef STRATA_SOURCE_DIR
#define STRATA_SOURCE_DIR "."   // (CI runs it from the source root)
#endif

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

std::string write_table(const std::filesystem::path& dir, const std::string& name, const std::string& header) {
    const std::filesystem::path path = dir / name;
    std::ofstream out(path);
    out << "# test table\n" << header << "\nf16 1536 4096 1536 1024 2390\nbf16 2048 512 2048 128 17\n";
    return path.string();
}

}  // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("hipblaslt_tuning_header_test_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    using strata::prefill::hipblaslt::TuningTable;
    std::string err;

    {
        TuningTable t;
        check(t.load(write_table(dir, "maya.txt", "MAYA_HIPBLASLT_TUNING_V1 gfx1151 100401"), "gfx1151", 100401, err),
              "MAYA_ header loads");
        check(t.rows().size() == 2 && t.arch() == "gfx1151" && t.version() == 100401, "MAYA_ header rows");
    }
    {
        TuningTable t;
        err.clear();
        check(t.load(write_table(dir, "legacy.txt", "STRATA_HIPBLASLT_TUNING_V1 gfx1151 100401"), "gfx1151", 100401,
                     err),
              "legacy STRATA_ header loads");
        check(t.rows().size() == 2, "legacy header rows");
    }
    for (const char* bad : {"MAYA_HIPBLASLT_TUNING_V2 gfx1151 100401", "HIPBLASLT_TUNING_V1 gfx1151 100401",
                            "STRATA_HIPBLASLT_TUNING_V1 gfx1151 100401 extra"}) {
        TuningTable t;
        err.clear();
        check(!t.load(write_table(dir, "bad.txt", bad), "gfx1151", 100401, err) &&
                  err.rfind("invalid tuning header", 0) == 0,
              bad);
    }
    {
        TuningTable t;
        err.clear();
        check(!t.load(write_table(dir, "arch.txt", "MAYA_HIPBLASLT_TUNING_V1 gfx1201 100401"), "gfx1151", 100401, err) &&
                  err.find("architecture mismatch") != std::string::npos,
              "architecture mismatch still rejected");
    }
    // every committed gfx1151 table loads against its own header's version
    for (const auto& entry : fs::directory_iterator(fs::path(STRATA_SOURCE_DIR) / "tools/hip")) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("gfx1151-", 0) != 0 || entry.path().extension() != ".txt") continue;
        std::ifstream in(entry.path());
        std::string line, magic, arch;
        int version = 0;
        while (std::getline(in, line))
            if (!line.empty() && line[0] != '#') break;
        std::istringstream(line) >> magic >> arch >> version;
        check(magic == "STRATA_HIPBLASLT_TUNING_V1", ("committed table keeps the legacy header: " + name).c_str());
        TuningTable t;
        err.clear();
        check(t.load(entry.path().string(), "gfx1151", version, err) && !t.rows().empty(),
              ("committed table loads: " + name + " " + err).c_str());
    }
    fs::remove_all(dir);
    if (failures) return 1;
    std::printf("hipblaslt tuning headers: MAYA_ and legacy STRATA_ accepted, others rejected, committed tables load\n");
    return 0;
}
