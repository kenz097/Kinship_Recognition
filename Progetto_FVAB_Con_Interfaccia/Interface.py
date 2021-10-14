#classe che contiene la struttuara dell'interfaccia grafica e il metodo main.


from PIL import ImageTk, Image
import tkinter as tk
import main_SIFT
import main_ReteNeurale

#Funzione per generare la finestra per inserire il path del video e il minutaggio di inizio gioco.
def MyButton_Elaboration():
    root_path = tk.Tk()
    root_path.title('Inserisci info video')
    root_path.geometry("480x250")
    label1 = tk.Label(root_path, text="Path Video", relief='flat')
    entry1 = tk.Entry(root_path, width=50)
    label2 = tk.Label(root_path, text="Minuto", relief='flat')
    entry2 = tk.Entry(root_path, width=25)
    b = tk.Button(root_path, text="ELABORA", command='', bg='green', fg='white', padx=10, pady=20)
    label1.pack(pady=10)
    entry1.pack()
    label2.pack(pady=10)
    entry2.pack()
    b.pack(pady=10)
    root_path.mainloop()

#Funzione per generare la tabella con i risultati del SIFT
def MyButton_SIFT():
    main_SIFT.functionSIFT()

#Funzione per generare la tabella con i risultati della rete neurale
def MyButton_Neural():
    main_ReteNeurale.functionNeural()



IMAGE_PATH = 'Soliti_ignoti_logo.jpg'
WIDTH, HEIGHT = 720, 480

root = tk.Tk()
root.title("Kinship Recognition")
root.geometry('{}x{}'.format(WIDTH, HEIGHT))

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT)
canvas.pack()

img = ImageTk.PhotoImage(Image.open(IMAGE_PATH).resize((WIDTH, HEIGHT), Image.ANTIALIAS))
canvas.background = img
bg = canvas.create_image(0, 0, anchor=tk.NW, image=img)

button1 = tk.Button(root, text="ELABORA VIDEO", command=MyButton_Elaboration, width=50,
                    bg='#ba0407', fg='#ffffff', relief='raised')
button_window1 = canvas.create_window(10, 10, anchor=tk.NW, window=button1)
button2 = tk.Button(root, text="NEURAL RESULT", command=MyButton_Neural, width=30,
                    bg='#ba0407', fg='#ffffff', relief='raised')
button_window2 = canvas.create_window(10, 60, anchor=tk.NW, window=button2)
button3 = tk.Button(root, text="SIFT RESULT", command=MyButton_SIFT, width=15,
                    bg='#ba0407', fg='#ffffff', relief='raised')
button_window3 = canvas.create_window(10, 110, anchor=tk.NW, window=button3)

root.mainloop()