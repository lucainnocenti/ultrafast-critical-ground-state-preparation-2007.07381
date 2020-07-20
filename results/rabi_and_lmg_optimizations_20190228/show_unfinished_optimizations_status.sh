for logfile in $(ls *log); do
	if [ ! -f ${logfile::-3}csv ]; then
		echo $logfile: $(grep -o "Starting iteration .*" $logfile | tail -1)
	fi
done
