from Tkinter import *
import tkFileDialog
#from . import analyze

class App(Frame):
	def __init__(self, master=None):
		Frame.__init__(self, master)
		self.pack()

		self.entrythingy = Entry()
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
		)
		self.create_menu()



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

	def open_files(self):
		files = tkFileDialog.askopenfilename(
			multiple=True
		)
		for f in files:
			print 'file: ', f


	def print_contents(self, event):
		print "hi. contents of entry is now ---->", \
		self.contents.get()

def init():
	# create the application
	myapp = App(Tk())

	#
	# here are method calls to the window manager class
	#
	myapp.master.title("My Do-Nothing Application")
	myapp.master.maxsize(1000, 400)

	# start the program
	myapp.mainloop()