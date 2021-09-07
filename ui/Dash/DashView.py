from PyQt5.QtWebEngineWidgets import QWebEngineView as QWebView
from PyQt5.QtCore import QUrl, QTimer

import uuid

# from Dash.DashApp import dashApp
from Dash.DashApp import dashApp, dashPageManager

import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objects as go
import plotly.express as px

from CounterFactualParameters import FeatureType

import pandas as pd

class DashView(QWebView):

    def __init__(self, parent=None):
        super(DashView, self).__init__()

        self.canvasID = str(uuid.uuid4())
        self.layout = None

        self.figure = None


    def updateGraph(self, parameters=None):
        if parameters is not None:
            dataframe = parameters['dataframe']
            model = parameters['model']
            xVariable = parameters['xVariable']
            yVariable = parameters['yVariable']

            # cleanDataframe = None
            # if xVariable != yVariable:
            #     cleanDataframe = dataframe[[xVariable, yVariable, 'distance', 'Class']]
            # else:
            #     cleanDataframe = dataframe[[xVariable, 'distance', 'Class']]
            # cleanDataframe = cleanDataframe.drop_duplicates()

            # fig = px.scatter(dataframe, x=xVariable, y=yVariable, color="species", size='petal_length', hover_data=['petal_width'])
            # self.figure = px.scatter(cleanDataframe, x=xVariable, y=yVariable, color='Class', size='distance')
            
            # APLICAR ENCODE NAS FEATURES CATEGÓRICAS
            dimensions = []

            for feature in dataframe.columns:   
                dictAux = {}             
                if feature == 'Class':
                    # dictAux['range'] = [dataframe[feature].min(), dataframe[feature].max()]
                    # dictAux['label'] = feature
                    # dictAux['values'] = dataframe[feature].to_numpy()
                    pass

                elif feature == 'distance':
                    dictAux['range'] = [dataframe[feature].min(), dataframe[feature].max()]
                    dictAux['label'] = feature
                    dictAux['values'] = dataframe[feature].to_numpy()

                else:
                    featureType = model.featuresInformations[feature]['featureType']

                    if featureType is FeatureType.Binary:
                        pass

                    elif featureType is FeatureType.Discrete or featureType is FeatureType.Numeric:
                        dataframe[feature] = pd.to_numeric(dataframe[feature])
                        dictAux['range'] = [dataframe[feature].min(), dataframe[feature].max()]
                        dictAux['label'] = feature
                        dictAux['values'] = dataframe[feature].to_numpy()

                    elif featureType is FeatureType.Categorical:
                        pass

                dimensions.append(dictAux)

            # self.figure = px.parallel_coordinates(dataframe)
            self.figure = go.Figure(data=
                go.Parcoords(
                    line_color='blue',
                    dimensions = dimensions
                )
            )

            self.layout = html.Div(
            dcc.Graph(
                id='graph',
                figure=self.figure)
            )

            dashApp.layout = self.layout
            dashPageManager.addPage(self.canvasID, self.layout)
            
            self.load(QUrl("http://127.0.0.1:8050" + "/" + self.canvasID))