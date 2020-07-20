for logfile in $(ls *log); do
	echo $logfile: $(grep -o "Starting iteration .*" $logfile | tail -1)
done