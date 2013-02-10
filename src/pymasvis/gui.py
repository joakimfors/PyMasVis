# -*- coding: utf-8 -*-
import os
import subprocess
import unicodedata
import tkFileDialog

from Tkinter import *
from threading import Thread
from . import analyze, widgets

FILETYPES=[('MP3', '*.mp3'), ('WAVE', '*.wav'), ('FLAC', '*.flac'), ('OGG', '*.ogg'), ('MP4', '*.mp4'), ('M4A', '*.m4a'), ('AIFF', '*.aiff'), ('AIF', '*.aif'), ('AU', '*.au'), ('SND', '*.snd')]

class App(Frame):
	def __init__(self, master=None):
		Frame.__init__(self, master)
		self.pack(fill=BOTH, expand=YES, anchor=W)

		"""self.entrythingy = Entry()
		self.entrythingy.pack()

		# here is the application variable
		self.contents = StringVar()
		# set it to some value
		self.contents.set("this is a variable")
		# tell the entry widget to watch this variable
		self.entrythingy["textvariable"] = self.contents

		# and here we get a callback when the user hits return.
		# we will have the program print out the value of the
		# application variable when the user hits return
		self.entrythingy.bind(
			'<Key-Return>',
			self.print_contents
		)"""
		self.create_menu()
		self.create_toolbar()
		self.create_fileview()
		self.create_statusbar()



	def hello(self):
		print "hello!"

	def create_menu(self):
		##
		## Menu
		##
		menubar = Menu(self)

		# create a pulldown menu, and add it to the menu bar
		filemenu = Menu(menubar, tearoff=0)
		filemenu.add_command(label="Open", command=self.open_files)
		#filemenu.add_command(label="Save", command=self.hello)
		filemenu.add_separator()
		filemenu.add_command(label="Exit", command=self.quit)
		menubar.add_cascade(label="File", menu=filemenu)

		# create more pulldown menus
		#editmenu = Menu(menubar, tearoff=0)
		#editmenu.add_command(label="Cut", command=self.hello)
		#editmenu.add_command(label="Copy", command=self.hello)
		#editmenu.add_command(label="Paste", command=self.hello)
		#menubar.add_cascade(label="Edit", menu=editmenu)

		#helpmenu = Menu(menubar, tearoff=0)
		#helpmenu.add_command(label="About", command=self.hello)
		#menubar.add_cascade(label="Help", menu=helpmenu)

		# display the menu
		self.master.config(menu=menubar)

	def create_toolbar(self):
		self.toolbar = Frame(self)
		self.analyze = Button(self.toolbar, text="Analyze", width=6, command=self.do_analyze)
		self.analyze.pack(side=LEFT, padx=2, pady=2)
		self.toolbar.pack(side=TOP, fill=X)

	def create_fileview(self):
		self.fileview = DDList(self,
			activestyle="none",
			selectmode=SINGLE
		)
		self.fileview.pack(fill=BOTH, expand=YES, anchor=W)

	def create_busybar(self):
		self.busybar = widgets.BusyBar(self, text='')
		self.busybar.pack(side=TOP, fill=X)
		self.busybar.on()

	def create_statusbar(self):
		self.status = StatusBar(self)
		self.status.pack(side=BOTTOM, fill=X)

	def open_files(self):
		files = tkFileDialog.askopenfilename(
			multiple=True,
			filetypes=FILETYPES
		)
		for f in files:
			print 'file: ', f
			self.fileview.insert(END, DDFile(unicodedata.normalize('NFC', f)))

	def do_analyze(self):
		files = self.fileview.get_selected()
		print "files to analyze: ", files
		self.analyze.config({'state':DISABLED})
		self.create_busybar()
		Thread(target=self.run_analyze, kwargs={'files': files}).start()

	def run_analyze(self, files):
		for f in files:
			print "analyze", f, f.filename
			result = analyze.analyze(f.filename)
			subprocess.check_call(["open", result])
		self.analyze.config({'state':NORMAL})
		self.busybar.destroy()

	def print_contents(self, event):
		print "hi. contents of entry is now ---->", \
		self.contents.get()

class StatusBar(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.label = Label(self, bd=1, relief=SUNKEN, anchor=W)
        self.label.pack(fill=X)

    def set(self, format, *args):
        self.label.config(text=format % args)
        self.label.update_idletasks()

    def clear(self):
        self.label.config(text="")
        self.label.update_idletasks()

class DDList(Listbox):
	""" A Tkinter listbox with drag'n'drop reordering of entries. """
	def __init__(self, master, **kw):
		self.items = []
		kw['selectmode'] = SINGLE
		Listbox.__init__(self, master, kw)
		self.bind('<Button-1>', self.set_current)
		self.bind('<B1-Motion>', self.shift_selection)
		self.bind('<Delete>', self.remove_selected)
		self.bind('<BackSpace>', self.remove_selected)
		self.curIndex = None

	def insert(self, index, item):
		Listbox.insert(self, index, item)
		print "insert at index: ", index, item
		if index == "end":
			self.items.append(item)
		elif index == "first":
			self.items.insert(0, item)
		else:
			self.items.insert(index, item)

	def delete(self, first, last=None):
		Listbox.delete(self, first, last)
		print "delete ", first, last
		if last:
			for i in range(first, last+1):
				self.items.pop(i)
		else:
			self.items.pop(first)

	def get(self, first, last=None):
		print "get ", first, last
		if last:
			return [self.items[i] for i in range(first, last+1)]
		else:
			return self.items[first]

	def remove_selected(self, event):
		indices = self.curselection()
		for index in indices:
			self.delete(int(index))

	def set_current(self, event):
		self.curIndex = self.nearest(event.y)

	def shift_selection(self, event):
		i = self.nearest(event.y)
		if i < self.curIndex:
			x = self.get(i)
			self.delete(i)
			self.insert(i+1, x)
			self.curIndex = i
		elif i > self.curIndex:
			x = self.get(i)
			self.delete(i)
			self.insert(i-1, x)
			self.curIndex = i

	def get_selected(self):
		items = self.curselection()
		print items
		return [self.get(int(item)) for item in items]

class DDFile():
	def __init__(self, filename):
		self.filename = filename

	def __str__(self):
		return os.path.basename(self.filename).encode("utf-8")

		"""Here is an example of use of this DDList class, presented, as usual, with a guard of if _ _name_ _ == ' _ _main_ _ ' so we can make it part of the module containing the class and have it run when the module is executed as a "main script":

		if _ _name_ _ == '_ _main_ _': tk = Tkinter.Tk( ) length = 10 dd = DDList(tk, height=length) dd.pack( ) for i in xrange(length): dd.insert(Tkinter.END, str(i)) def show( ): ''' show the current ordering every 2 seconds ''' for x in dd.get(0, Tkinter.END): print x, print tk.after(2000, show) tk.after(2000, show) tk.mainloop( )"""


def init():
	# create the application
	myapp = App()

	#
	# here are method calls to the window manager class
	#
	myapp.master.title("PyMasVis")
	myapp.master.geometry("400x600+40+40")
	#myapp.master.maxsize(1000, 400)

	# start the program
	myapp.mainloop()