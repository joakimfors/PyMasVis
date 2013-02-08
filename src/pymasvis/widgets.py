from Tkinter import *
import time

class BusyBar(Frame):
    def __init__(self, master=None, **options):
        # make sure we have sane defaults
        self.master=master
        self.options=options
        self.width=options.setdefault('width', 100)
        self.height=options.setdefault('height', 12)
        self.background=options.setdefault('background', None)
        self.relief=options.setdefault('relief', 'flat')
        self.bd=options.setdefault('bd', 1)

        #extract options not applicable to frames
        self._extractOptions(options)

        # init the base class
        Frame.__init__(self, master, options)

        self.incr=self.width*self.increment
        self.busy=0
        self.dir='right'

        # create the canvas which is the container for the bar
        self.canvas=Canvas(self, height=self.height, width=self.width, bd=0,
                           highlightthickness=0, background=self.background)
        # catch canvas resizes
        self.canvas.bind('<Configure>', self.onSize)

        # this is the bar that moves back and forth on the canvas
        self.scale=self.canvas.create_rectangle(0, 0, self.width*self.barWidth, self.height, fill=self.fill)

        # label that is in the center of the widget
        self.label=self.canvas.create_text(self.canvas.winfo_reqwidth() / 2,
                                           self.height / 2, text=self.text,
                                           anchor="c", fill=self.foreground,
                                           font=self.font)
        self.update()
        self.canvas.pack(side=TOP, fill=X, expand=NO)

    def _extractOptions(self, options):
        # these are the options not applicable to a frame
        self.foreground=pop(options, 'foreground', 'black')
        self.fill=pop(options, 'fill', 'grey')
        self.interval=pop(options, 'interval', 30)
        self.font=pop(options, 'font','helvetica 10')
        self.text=pop(options, 'text', '')
        self.barWidth=pop(options, 'barWidth', 0.5)
        self.increment=pop(options, 'increment', 0.05)

    # todo - need to implement config, cget, __setitem__, __getitem__ so it's more like a reg widget
    # as it is now, you get a chance to set stuff at the constructor but not after

    def onSize(self, e=None):
        self.width = e.width
        self.height = e.height
        # make sure the label is centered
        self.canvas.delete(self.label)
        self.label=self.canvas.create_text(self.width / 2, self.height / 2, text=self.text,
                                           anchor="c", fill=self.foreground, font=self.font)

    def on(self):
        self.busy = 1
        self.canvas.after(self.interval, self.update)

    def off(self):
        self.busy = 0

    def update(self):
        # do the move
        x1,y1,x2,y2 = self.canvas.coords(self.scale)
        if x2>=self.width:
            self.dir='left'
        if x1<=0:
            self.dir='right'
        if self.dir=='right':
            self.canvas.move(self.scale, self.incr, 0)
        else:
            self.canvas.move(self.scale, -1*self.incr, 0)

        if self.busy:
            self.canvas.after(self.interval, self.update)
        self.canvas.update_idletasks()

def pop(dict, key, default):
      value = dict.get(key, default)
      if dict.has_key(key):
          del dict[key]
      return value