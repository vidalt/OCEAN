import numpy as np

from matplotlib.artist import Artist
from matplotlib.lines import Line2D

from PyQt5.QtCore import pyqtSignal, QObject

class PolygonInteractor(QObject):
    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    # the updated current point values
    updatedPoint = pyqtSignal(object, list)

    def __init__(self, ax, poly, ranges, decimals, actionables, color):
        super(PolygonInteractor, self).__init__()
        
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure '
                               'or canvas before defining the interactor')
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly
        self.ranges = ranges
        self.decimals = decimals
        self.actionables = actionables
        self.color = color

        self._x, self._y = zip(*self.poly.xy)
        self.line = Line2D(self._x, self._y, color=self.color,
                           marker='o', markerfacecolor=self.color,
                           animated=True)
        self.ax.add_line(self.line)

        self.cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        canvas.mpl_connect('draw_event', self.on_draw)
        canvas.mpl_connect('button_press_event', self.on_button_press)
        canvas.mpl_connect('key_press_event', self.on_key_press)
        canvas.mpl_connect('button_release_event', self.on_button_release)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas = canvas

    def on_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        # do not need to blit here, this will fire before the screen is
        # updated

    def poly_changed(self, poly):
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def on_button_press(self, event):
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def on_button_release(self, event):
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None

        # emit the current point updated values
        x, y = zip(*self.poly.xy)
        self.updatedPoint.emit(self, list(y))

    def on_key_press(self, event):
        if not event.inaxes:
            return

        if self.line.stale:
            self.canvas.draw_idle()

    def on_mouse_move(self, event):
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        x, y = event.xdata, event.ydata

        if y < 0:
            y = 0
        elif y > self.ranges[self._ind]:
            y = self.ranges[self._ind]

        # feature not actionable 
        if not self.actionables[self._ind]:
            return
        
        # to use round function to drag categorical and integer values
        self.poly.xy[self._ind] = self._x[self._ind], round(y, self.decimals[self._ind])

        self.line.set_data(zip(*self.poly.xy))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)
        
