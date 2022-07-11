import dash
import flask
from urllib.parse import urlparse
import dash_html_components as html
import os


class DashPageManager:

    pages = {}

    def addPage(self, pageName, pageLayout):
        self.pages[pageName] = pageLayout

    def removePage(self, pageName, pageLayout):
        if(pageName in self.pages):
            del self.pages[pageName]

    def getPage(self, pageName):
        return self.pages[pageName]


def serverLayouts(layoutDict, currentPage):

    def getCurrentPage():
        if not flask.has_request_context():
            return html.Div()

        referer = flask.request.headers.get('Referer', '')
        path = urlparse(referer).path
        pages = path.split('/')

        if len(pages) < 2 or not pages[1]:
            return html.Div('Root Page')

        page = pages[1]

        if page in layoutDict:
            return layoutDict[page]

        return html.Div('Nothing ')

    return getCurrentPage


assetsPath = os.getcwd()
assetsPath = os.path.join(assetsPath, 'assets')

server = flask.Flask(__name__)
dashApp = dash.Dash(__name__, server=server, assets_folder=assetsPath)
dashServer = dashApp.server
dashPageManager = DashPageManager()
dashApp.config.suppress_callback_exceptions = True
dashApp.layout = html.Div()


def updateLayouts(currentPage):
    dashApp.layout = serverLayouts(dashPageManager.pages, currentPage)


def shutdownServer():
    func = flask.request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@server.route('/shutdown', methods=['POST'])
def shutdown():
    shutdownServer()
    return 'Server shutting down...'
