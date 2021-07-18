from PyQt5.QtWebEngineWidgets import QWebEngineView as QWebView
from PyQt5.QtCore import QUrl, QTimer

import uuid

# from Dash.DashApp import dashApp
from Dash.DashApp import dashApp, dashPageManager

import dash_core_components as dcc
import dash_html_components as html

import plotly.express as px

class DashView(QWebView):

    def __init__(self, parent=None):
        super(DashView, self).__init__()

        self.canvasID = str(uuid.uuid4())
        self.layout = None

        self.figure = None


    def updateGraph(self, parameters=None):
        if parameters is not None:
            dataframe = parameters['dataframe']
            xVariable = parameters['xVariable']
            yVariable = parameters['yVariable']

            cleanDataframe = None
            if xVariable != yVariable:
                cleanDataframe = dataframe[[xVariable, yVariable, 'distance', 'Class']]
            else:
                cleanDataframe = dataframe[[xVariable, 'distance', 'Class']]
            cleanDataframe = cleanDataframe.drop_duplicates()

            # fig = px.scatter(dataframe, x=xVariable, y=yVariable, color="species", size='petal_length', hover_data=['petal_width'])
            self.figure = px.scatter(cleanDataframe, x=xVariable, y=yVariable, color='Class', size='distance')

            self.layout = html.Div(
            dcc.Graph(
                id='graph',
                figure=self.figure)
            )

            dashApp.layout = self.layout
            dashPageManager.addPage(self.canvasID, self.layout)
            
            self.load(QUrl("http://127.0.0.1:8050" + "/" + self.canvasID))