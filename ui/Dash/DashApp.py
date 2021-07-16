
import dash

class DashPageManager:

    pages = {}

    def addPage(self,pageName, pageLayout):
        if(not(pageName in self.pages)):
            self.pages[pageName] = pageLayout

    def removePage(self,pageName, pageLayout):
        if(pageName in self.pages):
            del self.pages[pageName]
    
    def getPage(self,pageName):
        return self.pages[pageName]


# dashApp = dash.Dash(__name__, suppress_callback_exceptions=True)
dashApp = dash.Dash()
dashServer = dashApp.server
dashPageManager = DashPageManager()

