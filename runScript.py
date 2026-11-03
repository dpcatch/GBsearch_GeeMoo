import os
import subprocess
import multiprocessing


Ncpus = 16


config_path = 'data/search_params.json'
os.environ["LDC_CONFIG"] = config_path


def search(i, which_run):
	try:
		#core = i % Ncpus
		core = (multiprocessing.current_process()._identity[0] - 1) % Ncpus
		command = "taskset -c %d ldc-gb-search %s %d" % (core, which_run, i)
		subprocess.run([command], shell=True)
		pass
	except Exception as e:
		print(f"Exception in worker with args {args}:\n{traceback.format_exc()}")
		raise

def merge(which_run):
	command = "ldc-gb-merge %s" % which_run
	subprocess.run([command], shell=True)




#do a first run to identify the number of needed windows
N = 0
command = "taskset -c 0 ldc-gb-search even1st 0"
result = subprocess.run([command], capture_output=True, text=True, check=True, shell=True)

for line in result.stdout.splitlines():
        l = line.split("total number of windows ")
        if(len(l) > 1):
                N = int(l[1])

print(N)



#run batches in revers order for a better loadbalancing of the cpus
iterations = list(range(N, -1, -1))
#iterations = list(range(1816,1831))


for which_run in ["even1st", "odd", "even"]:
	with multiprocessing.Pool(processes=Ncpus) as pool:
		args_list = [(i, which_run) for i in iterations]
		pool.starmap(search, args_list, chunksize=1)
	merge(which_run)

merge("global")

