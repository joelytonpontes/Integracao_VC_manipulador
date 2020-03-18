#####################################################################################################################################
#####################################################################################################################################
#################################               UFPB-CEAR                                           #################################
#################################               Aluno: Joelyton de Pontes Silva                     #################################
#################################               Orientador: Ademar Virgolino                        #################################
#####################################################################################################################################
#####################################################################################################################################
########             Programa de ajuste de pesepctiva para integração de visão computacional a manipular robótico.           ########
#####################################################################################################################################


#utilizando a mesa (entre os parafusos) como delimitador da workspace
### proporção do retangulo = 2.157
### 2500 e 5393 # tamanho da imagem final seguindo a proporção
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_circle(event,x,y,flags,param):
    #desenha as circuferencias azuis ai dar dois clicks com o mouse, e salva as cordenadas desse ponto
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),5,(255,0,0),-1)
        ponto = [x, y]
        pontos.append(ponto)

def Camera():
    #utilização da câmera
    captura = cv2.VideoCapture(0) #instancia o uso da webcam
 
    while(1):
        ret, frame = captura.read() #pega efetivamente a imagem da webcam
           
        cv2.imshow('camera fixada? Aperte Esc para concluir',frame)
    
        k = cv2.waitKey(10) & 0xff #fecha a janela
        if k == 27:
            break
    
    captura.release() #dispensa o uso da webcam
    cv2.destroyAllWindows()
    return frame    #ultimo frame capturado   

img = Camera()
pontos= []

#img = cv2.imread("imagem/calibracao_2_17_01.jpg") #caso queira testar com uma imagem ao inves da câmera
img = cv2.resize(img, (639,479), interpolation = cv2.INTER_AREA)
print("selecione as 4 extremidades da imagem\nseguindo a ordem do exemplo abaixo\n ##########\n # 1    2 #\n # 4    3 #\n ##########")
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)#habilita a função draw_circle


while 1:
    #garante q haverá 4 pontos
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
    if len (pontos) == 4:
        break
cv2.destroyAllWindows()



rows,cols,ch = img.shape


pts1 = np.float32([pontos[0], pontos[1], pontos[3], pontos[2]])
##pts2 deve seguir a propporção do retangulo base densenhado pelo braço
## o retangulo já esta pre-desenhado em .cod na memoria do braço
pts2 = np.float32([[0,0], [5393,0], [0,2500], [5393,2500]])

M = cv2.getPerspectiveTransform(pts1,pts2)#distorção controlada para diminuir a distorção causada pelo angulo da câmera

dst = cv2.warpPerspective(img,M,(5393,2500))#aplicando na imagem



##converter pra plotar com plt
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst1 = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)


plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst1),plt.title('Output')
plt.show()

cv2.imwrite("editadas/teste_cam2.jpg", dst)
print ("pontos a serem setados no programa principal\npontos = ", pontos)