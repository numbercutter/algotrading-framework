import datetime as dt
import plotly.graph_objects as go
from pyparsing import line
from plotly.subplots import make_subplots

class CandlePlot:

    def __init__(self, df, candles=True, subplots=None):
        self.df_plot = df.copy()
        self.candles = candles
        self.create_candle_fig(subplots)


    def add_timestr(self):
        self.df_plot['sTime'] = [dt.datetime.strftime(x, "s%y-%m-%d %H:%M") 
                        for x in self.df_plot.time]

    def create_candle_fig(self, sub):
        self.add_timestr()
        
        self.fig = go.Figure()
        if sub != None:
            self.fig = make_subplots(rows=sub, cols=1, shared_xaxes=True)
        if self.candles == True:
            self.fig.add_trace(go.Candlestick(
                x=self.df_plot.sTime,
                open=self.df_plot.mid_o,
                high=self.df_plot.mid_h,
                low=self.df_plot.mid_l,
                close=self.df_plot.mid_c,
                line=dict(width=1), opacity=1,
                increasing_fillcolor='#FFFFFF',
                decreasing_fillcolor="#FFFFFF",
                increasing_line_color='#FFFFFF',  
                decreasing_line_color='#FFFFFF'
            ))

    def update_layout(self, width, height, nticks):
        self.fig.update_yaxes(
            gridcolor="#000000"
            )
        self.fig.update_xaxes(
            gridcolor="#5A5A5A",
            rangeslider=dict(visible=False),
            nticks=nticks
        )

        self.fig.update_layout(
            width=width,
            height=height,
            margin=dict(l=10,r=10,b=10,t=10),
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",
            font=dict(size=8, color="#e1e1e1")
        )

    def add_traces(self, line_traces):
        for t in line_traces:
            self.fig.add_trace(go.Scatter(
                x=self.df_plot.sTime,
                y=self.df_plot[t],
                line=dict(width=2),
                line_shape="spline",
                name=t
            ))

    def add_h_line(self, line):
        self.fig.add_hline(y=line, line_color="pink")

    def add_h(self, lines):

        for v in lines:
            l = len(v)
            if l > 1:
                self.fig.add_hrect(
                    y0=v[0], y1=v[-1], line_width=0, 
                    fillcolor="seagreen", opacity=0.7)
            else:
                self.fig.add_hrect(
                    y0=v[0]-0.0002, y1=v[0]+0.0002, line_width=0, 
                    fillcolor="seagreen", opacity=0.7)
            
    def add_sub_traces(self, r=2, c=1):
        self.fig.add_trace(
            go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
            row=r, col=c
        )

    def show_plot(self, width=900, height=400, nticks=5, line_traces=[], lines=[]):
        self.add_traces(line_traces)
        self.add_h(lines)
        self.update_layout(width, height, nticks)
        self.fig.show()

    def plot_img(self, width=900, height=400, nticks=5, line_traces=[], lines=[], filename='plot.png'):
        self.add_traces(line_traces)
        self.add_h(lines)
        self.update_layout(width, height, nticks)
        self.fig.write_image(f'images/{filename}', scale=2)