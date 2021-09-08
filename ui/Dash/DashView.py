from PyQt5.QtWebEngineWidgets import QWebEngineView as QWebView
from PyQt5.QtCore import QUrl

import uuid

from sklearn.preprocessing import LabelEncoder

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
            dataframe['Class'] = pd.to_numeric(dataframe['Class'])
            dimensions = []

            for feature in dataframe.columns:   
                dictAux = {}             
                if feature == 'Class':
                    dataframe[feature] = pd.to_numeric(dataframe[feature])
                    dictAux['range'] = [0, 2]
                    dictAux['label'] = feature
                    dictAux['values'] = dataframe[feature].to_numpy()
                    dictAux['tickvals'] = [0, 1, 2]
                    dictAux['ticktext'] = ['0', '1', 'current']

                elif feature == 'distance':
                    dataframe[feature] = pd.to_numeric(dataframe[feature])
                    dictAux['range'] = [dataframe[feature].min(), dataframe[feature].max()]
                    dictAux['label'] = feature
                    dictAux['values'] = dataframe[feature].to_numpy()

                else:
                    featureType = model.featuresInformations[feature]['featureType']

                    if featureType is FeatureType.Binary or featureType is FeatureType.Categorical:
                        uniqueValues = dataframe[feature].unique()
                        encoder = LabelEncoder()
                        encoder.fit(uniqueValues)

                        dataframe[feature] = encoder.transform(dataframe[feature])

                        dictAux['range'] = [dataframe[feature].min(), dataframe[feature].max()]
                        dictAux['label'] = feature
                        dictAux['values'] = dataframe[feature].to_numpy()
                        dictAux['tickvals'] = encoder.transform(uniqueValues)
                        dictAux['ticktext'] = uniqueValues

                    elif featureType is FeatureType.Discrete or featureType is FeatureType.Numeric:
                        dataframe[feature] = pd.to_numeric(dataframe[feature])
                        dictAux['range'] = [dataframe[feature].min(), dataframe[feature].max()]
                        dictAux['label'] = feature
                        dictAux['values'] = dataframe[feature].to_numpy()

                dimensions.append(dictAux)

            # self.figure = px.parallel_coordinates(dataframe)
            self.figure = go.Figure(data=
                go.Parcoords(
                    line = dict(color = dataframe['Class'],
                    # colorscale = px.colors.sequential.Viridis,
                    colorscale = [(0.00, "red"), (0.33, "red"), 
                                  (0.33, "green"), (0.66, "green"),
                                  (0.66, "blue"),  (1.00, "blue")],
                    showscale = True,
                    cmin = dataframe['Class'].min(),
                    cmax = dataframe['Class'].max()),
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

    def updateFeatureImportanceGraph(self, parameters=None):
        if parameters is not None:
            dataframe = parameters['dataframe']
            xVariable = parameters['xVariable']
            yVariable = parameters['yVariable']

            self.figure = px.bar(dataframe, x=xVariable, y=yVariable)

            self.layout = html.Div(
            dcc.Graph(
                id='graph',
                figure=self.figure)
            )

            dashApp.layout = self.layout
            dashPageManager.addPage(self.canvasID, self.layout)
            
            self.load(QUrl("http://127.0.0.1:8050" + "/" + self.canvasID))
