import torch.multiprocessing as mp

class Async(mp.Process):
    __ProcessID = 0
    ProcessDict = dict()
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self)
        self._queue = mp.Queue(maxsize=10)
        self._worker_queue = mp.Queue(maxsize=10)
        self.__process_id = Async.__ProcessID
        Async.ProcessDict[self.__process_id] = self
        Async.__ProcessID+=1

    def _send(self, msg=None):
        self._worker_queue.put(msg)

    def _receive(self):
        (cmd, msg) = self._queue.get()
        return cmd, msg

    def send(self, cmd=None, msg=None):
        self._queue.put([cmd,msg])

    def receive(self):
        msg = self._worker_queue.get()
        return msg

    def _get_id(self):
        return self.__process_id