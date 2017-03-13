import numpy as np
import matplotlib.pyplot as plt

class Line:

    def __init__(self):
        self.__tracking_count   = 10   # Number of previous fits to track
        self.__recent_xfitted   = []   # x values of the last N fits of the line
        self.__recent_ploty     = []   # y values of the last N fits of the line
        self.__best_fit         = None # polynomial coefficients averaged over the last N iterations
        self.__frame_drop_count = 0    # Count dropped frames because of bad detection
        self.__frame_drop_limit = 5    # Frame drop limit to search histogram again
        self.histogram_search   = True # Use histpgram method if True
        
    def save_fit(self, recent_xfit, ploty):
        
        if len(self.__recent_xfitted) == self.__tracking_count:
            del self.__recent_xfitted[0]
            del self.__recent_ploty[0]

        self.__recent_xfitted.append(recent_xfit)
        self.__recent_ploty.append(ploty)

    def last_fitx(self, ploty):
        if self.__best_fit is None:
            self.best_fit(ploty)

        return self.__best_fit[0]*ploty**2 + self.__best_fit[1]*ploty + self.__best_fit[2]

    def sanity(self, s):
        if s == False:
            self.__frame_drop_count += 1
            if self.__frame_drop_count == self.__frame_drop_limit:
                print("Too many frames dropped - Searching histogram again")
                self.histogram_search = True
                self.__frame_drop_count = 0
        else:
            self.__frame_drop_count = 0 

    def best_fit(self, ploty):
        x = np.array(self.__recent_xfitted[0:]).ravel()
        y = np.array(self.__recent_ploty[0:]).ravel()

        self.__best_fit = np.polyfit(y, x, 2)
        return self.__best_fit


