import time
import threading

import numpy as np

import vtk
import vtk.util.numpy_support as vtk_np


import pyrealsense as pyrs
serv = pyrs.Service()
serv.start()
cam = serv.Device()


class VTKActorWrapper(object):
    def __init__(self, nparray):
        super(VTKActorWrapper, self).__init__()

        self.nparray = nparray

        nCoords = nparray.shape[0]
        nElem = nparray.shape[1]

        self.verts = vtk.vtkPoints()
        self.cells = vtk.vtkCellArray()
        self.scalars = None

        self.pd = vtk.vtkPolyData()
        self.verts.SetData(vtk_np.numpy_to_vtk(nparray))
        self.cells_npy = np.vstack([np.ones(nCoords,dtype=np.int64),
                               np.arange(nCoords,dtype=np.int64)]).T.flatten()
        self.cells.SetCells(nCoords,vtk_np.numpy_to_vtkIdTypeArray(self.cells_npy))
        self.pd.SetPoints(self.verts)
        self.pd.SetVerts(self.cells)

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputDataObject(self.pd)

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetRepresentationToPoints()
        self.actor.GetProperty().SetColor(0.0,1.0,0.0)

    def update(self, threadLock, update_on):
        thread = threading.Thread(target=self.update_actor, args=(threadLock, update_on))
        thread.start()

    def update_actor(self, threadLock, update_on):
        while (update_on.is_set()):
            time.sleep(0.01)
            threadLock.acquire()
            cam.wait_for_frames()
            self.nparray[:] = cam.points.reshape(-1,3)
            self.pd.Modified()
            threadLock.release()


class VTKVisualisation(object):
    def __init__(self, threadLock, actorWrapper, axis=True,):
        super(VTKVisualisation, self).__init__()

        self.threadLock = threadLock

        self.ren = vtk.vtkRenderer()
        self.ren.AddActor(actorWrapper.actor)

        self.axesActor = vtk.vtkAxesActor()
        self.axesActor.AxisLabelsOff()
        self.axesActor.SetTotalLength(1, 1, 1)
        self.ren.AddActor(self.axesActor)

        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.ren)

        ## IREN
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)
        self.iren.Initialize()

        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(self.style)

        self.iren.AddObserver("TimerEvent", self.update_visualisation)
        dt = 30 # ms
        timer_id = self.iren.CreateRepeatingTimer(dt)

    def update_visualisation(self, obj=None, event=None):
        time.sleep(0.01)
        self.threadLock.acquire()
        self.ren.GetRenderWindow().Render()
        self.threadLock.release()


def main():
    update_on = threading.Event()
    update_on.set()

    threadLock = threading.Lock()

    cam.wait_for_frames()
    pc = cam.points.reshape(-1,3)
    actorWrapper = VTKActorWrapper(pc)
    actorWrapper.update(threadLock, update_on)

    viz = VTKVisualisation(threadLock, actorWrapper)
    viz.iren.Start()
    update_on.clear()


main()
cam.stop()
serv.stop()
