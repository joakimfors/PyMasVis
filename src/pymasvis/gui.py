import os
import tkFileDialog

from Tkinter import *
#from . import analyze

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
		self.create_fileview()



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
		filemenu.add_command(label="Save", command=self.hello)
		filemenu.add_separator()
		filemenu.add_command(label="Exit", command=self.quit)
		menubar.add_cascade(label="File", menu=filemenu)

		# create more pulldown menus
		editmenu = Menu(menubar, tearoff=0)
		editmenu.add_command(label="Cut", command=self.hello)
		editmenu.add_command(label="Copy", command=self.hello)
		editmenu.add_command(label="Paste", command=self.hello)
		menubar.add_cascade(label="Edit", menu=editmenu)

		helpmenu = Menu(menubar, tearoff=0)
		helpmenu.add_command(label="About", command=self.hello)
		menubar.add_cascade(label="Help", menu=helpmenu)

		# display the menu
		self.master.config(menu=menubar)

	def create_fileview(self):
		self.fileview = DDList(self,
			activestyle="none",
			selectmode=SINGLE
		)
		self.fileview.pack(fill=BOTH, expand=YES, anchor=W)


	def open_files(self):
		files = tkFileDialog.askopenfilename(
			multiple=True
		)
		for f in files:
			print 'file: ', f
			self.fileview.insert(END, DDFile(f))


	def print_contents(self, event):
		print "hi. contents of entry is now ---->", \
		self.contents.get()

class DDList(Listbox):
	""" A Tkinter listbox with drag'n'drop reordering of entries. """
	def __init__(self, master, **kw):
		kw['selectmode'] = SINGLE
		Listbox.__init__(self, master, kw)
		self.bind('<Button-1>', self.setCurrent)
		self.bind('<B1-Motion>', self.shiftSelection)
		self.curIndex = None

	def setCurrent(self, event):
		self.curIndex = self.nearest(event.y)

	def shiftSelection(self, event):
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

class DDFile():
	def __init__(self, filename):
		self.filename = filename

	def __str__(self):
		return os.path.basename(self.filename)

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