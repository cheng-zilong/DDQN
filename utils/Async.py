import torch.multiprocessing as mp

class Async(mp.Process):
    __ProcessID = 0
    ProcessDict = dict()
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self)
        self._pipe, self._worker_pipe = mp.Pipe()
        self.__process_id = Async.__ProcessID
        Async.ProcessDict[self.__process_id] = self
        Async.__ProcessID+=1

    def _send(self, msg=None):
        self._worker_pipe.send(msg)

    def _receive(self):
        (cmd, msg) = self._worker_pipe.recv()
        return cmd, msg

    def send(self, cmd=None, msg=None):
        self._pipe.send([cmd,msg])

    def receive(self):
        msg = self._pipe.recv()
        return msg

    def _get_id(self):
        return self.__process_id