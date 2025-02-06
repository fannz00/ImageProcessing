from multiprocessing import Process, Queue, Value
from queue import Empty
import time
import cProfile


class ProcessPool:
    """
    A simple process pool for executing tasks in parallel using multiple processes.

    This class manages a pool of worker processes that can execute a given function
    on tasks fetched from a queue. It allows control over task addition, starting
    and stopping of processes, and checking if the pool is still running.

    Attributes:
        func (callable): The function to be executed by each worker process on the tasks.
        tasks (Queue): A queue holding tasks to be processed by the workers.
        running (multiprocessing.Value): A shared value indicating whether the pool is active.
        processes (list): A list of worker processes managed by the pool.

    Methods:
        add_task(task): Adds a task to the task queue.
        worker(index, name): The worker process function that processes tasks.
        start(n_processes, name): Starts the specified number of worker processes.
        stop(slow): Stops the worker processes either gracefully or immediately.
        is_running(): Checks if any worker processes are still running.
    """
    def __init__(self, func, running, max_tasks: int = 10) -> None:
        """
        Initialize the ProcessPool with a function, running flag, and optional max tasks.

        Args:
            func (callable): The function to execute for each task in the pool.
            running (multiprocessing.Value): A shared flag indicating if the pool is active.
            max_tasks (int, optional): Maximum number of tasks in the queue. Default is 10.
                                       If set to 0 or less, the queue size is unlimited.
        """
        self.func = func
        if max_tasks > 0:
            self.tasks = Queue(max_tasks)
        else:
            self.tasks = Queue()
        self.running = running

    def add_task(self, task):
        """
        Add a task to the task queue for processing by the worker processes.

        Args:
            task: The task to be added to the queue.
        """
        self.tasks.put(task)

    def worker(self, index: int, name=''):
        """
        The worker function that runs in each process, fetching and executing tasks.

        This function runs in a loop, fetching tasks from the queue and executing
        them using the specified function. It stops when the running flag is set to 0
        or when a "quit" task is received.

        Args:
            index (int): The index of the worker process for identification.
            name (str, optional): An optional name prefix for the worker process.
        """
        # Create a profiler for each worker process
        # profiler = cProfile.Profile()
        # profiler.enable()

        while self.running.value == 1:
            try:
                task = self.tasks.get(timeout=1)
                if task == "quit":
                    self.running.value = 0
                    break
                self.func(task, index)
            except Empty:
                pass #muss hier vllt ein continue hin?
        
        # Disable profiling and dump stats to a file
        # profiler.disable()
        # profile_filename = f"profile_{name}_{index}.prof"
        # profiler.dump_stats(profile_filename)

        # print(f"{name} Process {index} finished and profile saved to {profile_filename}")

    def start(self, n_processes, name=''):
        """
        Start the specified number of worker processes.

        This method initiates a number of processes to begin executing tasks
        from the queue.

        Args:
            n_processes (int): The number of worker processes to start.
            name (str, optional): An optional name prefix for the worker processes.
        """
        self.processes = []

        for i in range(n_processes):
            p = Process(target=self.worker, args=(i,name))
            p.start()
            self.processes.append(p)

    def stop(self, slow: bool = False):
        """
        Stop the worker processes either gracefully or immediately.

        Args:
            slow (bool, optional): If True, add a "quit" task to the queue for graceful shutdown.
                                   If False, set the running flag to 0 to stop immediately.
        """
        if slow:
            self.add_task("quit")
        else:
            self.running.value = 0

    def is_running(self):
        """
        Check if any worker processes are still running.

        Returns:
            bool: True if any worker processes are still active, False otherwise.
        """
        if self.running.value == 1:
            return True
        else:
            for p in self.processes:
                if p.is_alive():
                    return True
            return False


if __name__ == "__main__":
    import time

    def f(x, i):
        print(f"Process {i} has the task {x}")
        time.sleep(1)

    pool = ProcessPool(f, 100)
    for i in range(100):
        pool.add_task(i)

    pool.start(5)
    pool.stop(True)
