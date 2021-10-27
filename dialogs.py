import tkinter as tk
import tkinter.filedialog as tkdialogs

def wait_widget(WidgetClass):
    """Decorates a widget class with local event loop and returns the
    instances result attribute on destruction.
    
    Assumptions:
    widget instance has a 'parent' attribute
    widget instance builds a 'result' attribute
    """

    def decorated(*args, **kwargs):
        #create instance and wait on parent
        instance = WidgetClass(*args, **kwargs)
        instance.parent.wait_window()
        #return instances result attr
        return instance.result
    return decorated

def root_deco(dialog):
    """Decorates a dialog with a toplevel that is destroyed when the dialog
    is closed."""

    def decorated(*args, **kwargs):
        #create root and withdraw from screen
        root = tk.Tk()
        root.withdraw()
        #open dialog returning result and destroy root
        result = dialog(*args, parent=root, **kwargs)
        root.destroy()
        return result
    return decorated

@root_deco
def standard_dialog(kind, **options):
    """Opens a standard tkinter modal file dialog and returns a result.

    Args:
        kind (str):             name of a tkinter dialog
    **options:
        parent (widget):        ignored
        title (str):            title of the dialog window
        initialdir (str):       dir dialog starts in 
        initialfile (str):      file selected on dialog open
        filetypes (seq):        sequence of (label, pattern tuples) '*'
                                wildcard allowed
        defaultextension (str): default ext to append during save dialogs
        multiple (bool):        when True multiple selection enabled
    """

    return getattr(tkdialogs, kind)(**options)

if __name__ == '__main__':

    #path = standard_dialog('askopenfilename', title='hubbub')

    path = standard_dialog('asksaveasfilename', defaultextension='.pkl')

