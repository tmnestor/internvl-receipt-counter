nohup python main.py --config config/config.yaml --mode train > training_log.txt 2>&1 &

# View the entire log file
  cat training_log.txt

  # View the last 20 lines
  tail -20 training_log.txt

  # View the log in real-time as it updates (most useful)
  tail -f training_log.txt

  # Press Ctrl+C to exit the tail -f view

  The tail -f command is particularly useful as it will continuously show new lines as they're added to the log file,
  allowing you to monitor training progress in real-time.

  You can also check if the process is still running using:

  # Find your Python process
  ps aux | grep "[p]ython main.py"

  # Or get the process ID from the nohup.out file that was created
  cat nohup.out
