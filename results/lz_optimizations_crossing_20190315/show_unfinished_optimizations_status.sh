for logfile in $(ls *log); do
	if [ ! -f ${logfile::-3}csv ]; then
		echo $logfile: $(grep -o "Iteration .*" $logfile | tail -1)
	fi
done
