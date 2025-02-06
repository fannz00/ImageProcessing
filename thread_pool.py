import threading
import time
import cProfile
import os

class ThreadPool:
    """
    A simple thread pool for executing tasks concurrently using multiple threads.

    This class manages a pool of worker threads that can execute a specified function
    on tasks stored in a task list. It provides mechanisms to add tasks, start and
    stop the threads, and check if the pool is still running.

    Attributes:
        SLOW_STOP_KEY (str): A special key used to signal a slow stop of the pool.
        todo (list): A list holding tasks to be processed by the worker threads.
        lock (threading.Lock): A lock to ensure thread-safe access to the task list.
        threads (list): A list of worker threads managed by the pool.
        run (bool): A flag indicating whether the pool is currently running.
        func (callable): The function to be executed by each worker thread on the tasks.
        max_todo_len (int): The maximum number of tasks allowed in the task list.
        max_sleep_counter (int): The maximum number of sleep cycles before a thread quits due to inactivity.

    Methods:
        worker(index): The worker thread function that processes tasks from the list.
        start(n_threads): Starts the specified number of worker threads.
        add_task(job): Adds a task to the task list for processing by the worker threads.
        stop(slow): Stops the worker threads either gracefully or immediately.
        is_running(): Checks if any worker threads are still running.
    """

    SLOW_STOP_KEY = "stop_pool"

    def __init__(self, func, max_todo_len: int = 10) -> None:
        """
        Initialize the ThreadPool with a function and optional maximum task length.

        Args:
            func (callable): The function to execute for each task in the pool.
            max_todo_len (int, optional): Maximum number of tasks in the task list. Default is 10.
        """
        self.todo = []
        self.lock = threading.Lock()
        self.threads = []
        self.run = True
        self.func = func
        self.max_todo_len = max_todo_len
        self.max_sleep_counter = 10000000

    def worker(self, index: int):
        """
        The worker function that runs in each thread, fetching and executing tasks.

        This function runs in a loop, fetching tasks from the task list and executing
        them using the specified function. It stops when the run flag is set to False
        or when the SLOW_STOP_KEY is encountered in the task list.

        Args:
            index (int): The index of the worker thread for identification.
        """
        # Create a profiler for each worker process
        profiler = cProfile.Profile()
        profiler.enable()

        sleep_counter = 0
        while True:
            if sleep_counter > self.max_sleep_counter or not self.run:
                print(f"Reader {index} quitting due to inactivity")
                break
            self.lock.acquire()
            if len(self.todo) == 0:
                self.lock.release()
                sleep_counter += 1
                time.sleep(0.01)
            else:
                sleep_counter = 0
                job = self.todo.pop(0)
                if job == self.SLOW_STOP_KEY:
                    self.run = False
                    print(f"Reader {index} stopped")
                    break
                self.lock.release()
                self.func(job, index)
        
        # Disable profiling and dump stats to a file
        profiler.disable()
        profile_filename = f"profile_thread_{index}.prof"
        profiler.dump_stats(profile_filename)
        print(f"Thread {index} finished and profile saved to {profile_filename}")

        #print(f"Reader {index} quitting")

    def start(self, n_threads: int):
        """
        Start the specified number of worker threads.

        This method initiates a number of threads to begin executing tasks
        from the task list.

        Args:
            n_threads (int): The number of worker threads to start.
        """
        for i in range(n_threads):
            t = threading.Thread(target=self.worker, args=(i,))
            t.start()
            self.threads.append(t)

    def add_task(self, job):
        """
        Add a task to the task list for processing by the worker threads.

        This method ensures thread-safe addition of tasks to the list, waiting
        if the list is full until a slot becomes available.

        Args:
            job: The task to be added to the list.
        """
        while True:
            self.lock.acquire()
            if len(self.todo) < self.max_todo_len:
                self.todo.append(job)
                self.lock.release()
                break
            self.lock.release()
            time.sleep(0.01)

    def stop(self, slow: bool = False):
        """
        Stop the worker threads either gracefully or immediately.

        Args:
            slow (bool, optional): If True, add a SLOW_STOP_KEY to the task list for graceful shutdown.
                                   If False, set the run flag to False to stop immediately.
        """
        if slow:
            self.todo.append(self.SLOW_STOP_KEY)
        else:
            self.run = False

    def is_running(self):
        """
        Check if any worker threads are still running.

        Returns:
            bool: True if any worker threads are still active, False otherwise.
        """
        if self.run:
            return True
        else:
            for t in self.threads:
                if t.is_alive():
                    return True
            return False
