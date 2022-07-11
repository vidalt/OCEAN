import uuid
from PyQt5 import QtWebEngineWidgets
from PyQt5.QtCore import QUrl
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
# Load UI functions
from .DashApp import dashApp, dashPageManager


class DashView(QtWebEngineWidgets.QWebEngineView):

    def __init__(self, parent=None):
        super(DashView, self).__init__()
        self.canvasID = str(uuid.uuid4())
        self.layout = None
        self.figure = None

    def updateFeatureImportanceGraph(self, dataframe, xVariable, yVariable):
        self.figure = px.bar(dataframe, x=xVariable, y=yVariable)
        self.layout = html.Div(
            dcc.Graph(
                id='graph',
                figure=self.figure)
            )
        dashApp.layout = self.layout
        dashPageManager.addPage(self.canvasID, self.layout)
        self.figure.update_layout(hoverlabel=dict(bgcolor="white"),
                                  font=dict(size=10),
                                  legend_font=dict(size=10),
                                  legend_title_font=dict(size=10),)
        # Set figure margins
        figlayout = self.figure.layout
        figlayout.margin.r = 20
        figlayout.margin.l = 5
        figlayout.margin.b = 5
        figlayout.margin.t = 70

        self.load(QUrl("http://127.0.0.1:8050" + "/" + self.canvasID))
