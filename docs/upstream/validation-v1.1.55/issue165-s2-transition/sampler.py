import glob,json,os,pathlib,select,sys,time
os.nice(10)
me=pathlib.Path('/proc')/str(os.getpid())
print(json.dumps({'sampler_identity':{'pid':os.getpid(),'start':(me/'stat').read_text().rsplit(')',1)[1].split()[19],'exe':os.readlink(me/'exe')},'hz':os.sysconf('SC_CLK_TCK'),'page_size':os.sysconf('SC_PAGE_SIZE'),'config':DATA}),flush=True)
def read(path):
 try:return pathlib.Path(path).read_text().strip()
 except OSError:return None
def fields(path,names):
 text=read(path)
 if text is None:return {}
 out={}
 for line in text.splitlines():
  parts=line.replace(':',' ').split()
  if parts and parts[0] in names:
   try:out[parts[0]]=int(parts[1])
   except (ValueError,IndexError):pass
 return out
def stat(path):
 text=read(path)
 if not text:return None
 v=text.rsplit(')',1)[1].split()
 out={'state':v[0],'minflt':int(v[7]),'majflt':int(v[9]),'utime':int(v[11]),'stime':int(v[12]),'threads':int(v[17]),'start':v[19],'vsize':int(v[20]),'rss_pages':int(v[21])}
 if len(v)>39:out.update(last_cpu=int(v[36]),delayacct_blkio_ticks=int(v[39]))
 return out
io_names={'rchar','wchar','syscr','syscw','read_bytes','write_bytes','cancelled_write_bytes'}
while not select.select([sys.stdin],[],[],1.)[0]:
 row={'time':time.time(),'monotonic':time.monotonic()}
 for pattern in ['/sys/class/drm/card0/device/pp_dpm_sclk','/sys/class/drm/card0/device/gpu_busy_percent','/sys/class/drm/card0/device/hwmon/hwmon*/temp1_input','/sys/class/drm/card0/device/hwmon/hwmon*/power1_average']:
  for name in glob.glob(pattern):
   value=read(name)
   if value is not None:row[name]=value
 row['pressure']={kind:read('/proc/pressure/'+kind) for kind in ('cpu','memory','io')}
 row['loadavg']=read('/proc/loadavg')
 row['memory_kib']=fields('/proc/meminfo',{'MemAvailable','MemFree','Cached','SwapTotal','SwapFree','Dirty','Writeback','AnonPages','Slab','PageTables'})
 row['vm']=fields('/proc/vmstat',{'pgfault','pgmajfault','pswpin','pswpout','pgscan_direct','pgscan_kswapd','pgsteal_direct','pgsteal_kswapd','allocstall_normal','allocstall_movable'})
 system_stat=read('/proc/stat') or ''
 row['cpu_ticks']=[int(x) for x in next((line for line in system_stat.splitlines() if line.startswith('cpu ')),'cpu').split()[1:]]
 row['system_counts']=fields('/proc/stat',{'ctxt','processes','procs_running','procs_blocked'})
 try:
  owner=json.loads(pathlib.Path(DATA['owner_file']).read_text())
  if DATA['rank']==0:pid=owner['rank0_pid'];start=owner['rank0_start'];exe=owner['rank0_exe']
  else:pid=owner['pid'];start=owner['start'];exe=owner['exe']
  proc=pathlib.Path('/proc')/str(pid);status=stat(proc/'stat')
  if status and status['start']==start and os.readlink(proc/'exe')==exe:
   row['owned']={'pid':pid,'stat':status,'io':fields(proc/'io',io_names),'status':fields(proc/'status',{'VmRSS','VmHWM','RssAnon','RssFile','RssShmem','Threads','voluntary_ctxt_switches','nonvoluntary_ctxt_switches'}),'schedstat':read(proc/'schedstat'),'threads':{}}
   for thread in (proc/'task').iterdir():
    t=stat(thread/'stat')
    if t:row['owned']['threads'][thread.name]={'stat':t,'schedstat':read(thread/'schedstat'),'io':fields(thread/'io',io_names)}
  else:row['owned_status']='recorded identity exited or reused'
 except (OSError,ValueError,KeyError):row['owned_status']='owned record not yet available'
 print(json.dumps(row,separators=(',',':')),flush=True)
