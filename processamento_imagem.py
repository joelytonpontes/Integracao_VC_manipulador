#####################################################################################################################################
#####################################################################################################################################
##################################               UFPB-CEAR                                          #################################
##################################               Aluno: Joelyton de Pontes Silva                    #################################
##################################               Orientador: Ademar Virgolino                       #################################
#####################################################################################################################################
#####################################################################################################################################
########          Programa de processamento de imagem para integração de visão computacional a manipular robótico.           ########
#####################################################################################################################################

#utilizando a mesa (entre os parafusos) como delimitador da workspace

#Bibliotecas (todas podem ser instaladas com auxilio do pip)
from cv2 import cv2 # opencv (pip install opencv-python)
import numpy as np
import Tracker_HSV as tr # as barrinnha iterativas que geram o filtro de cores (pip install Tracker-HSV)
import math
import pandas as pd
import matplotlib.pyplot as plt
from socket import *
import struct
import time

#######SET UP ########
#parâmetros de enquadramento
#se a camera for fixa, não há necessidade de ajustes dos pontos... porém, a situação tual, é necessario a recalibração periódica
##pontos = só pegar do programa Joe_ajuste_pespectiva e colar aqui
pontos =  [[156, 55], [545, 53], [563, 220], [139, 222]]
#(altura = x, largura = y) 
# imagem = cv braço= b up =U down = D 
#posicione a garra com angulo A = 0 no ponto 1
set_calibracao = [100, 782] #se em todos as tentativas de pegar as peças o erro for igual, se faz necessário alterar esse ponto de forma empírica
#se o braço se comportar como se a workspace fosse de tamanho diferente do que ralmente é, é necessario ajustar as linhas abaixo
bU=(set_calibracao[0] + 350), (set_calibracao[1] ) #no eixo bx 350 é aproximadamente o tamanho da área de atuação no eito bx
bD=(set_calibracao[0] ), (set_calibracao[1] - 740) #no eixo by 740 é aproximadamente o tamanho da área de atuação no eito by

#### A porporção é o eixo by/bx
proporcao=2.1575
cvU = [2500, int(2500*proporcao)]
cvD = [0, 0]
#delimitando area minima
maxi = 10000
tabela= [] # dataframe

#set TCP IP
socket_i=0
data=bytearray([0,0,0,0,0,0,0,0,0,0])
Neg=0 # flag, marca os numeros negativos
test=1 #flag de envio


#funções exclusivas de Varredura
def Angulo(ponto_a, ponto_b, center, img):
    #esse angulo é o formado entre a face e o eixo altura da imagem
    ponto_med = (((ponto_a[0] + ponto_b[0])/2),((ponto_a[1] + ponto_b[1])/2))
    angulo = math.atan((center[1]-ponto_med[1])/(-center[0]+ponto_med[0])) #tangente
    angulo = math.degrees(angulo)
    angulo = round(angulo,2)
    
    #bolinha azul que marcar a face que a garra vai pegar
    bolinha = (int(ponto_med[0]), int(ponto_med[1])) #marcar a face que a garra vai pegar
    cv2.circle(img,bolinha,15,(255,0,0),-1)

    return angulo
def Conversao_centro(center):
    #converte os valores em pixels para valores reais que o braço entende
    ponto_cv = [round(center[1],3), round(center[0],3)]
    ponto_b = [0, 0]
    #interpolação
    ponto_b[0] = -((((cvU[0]-ponto_cv[0])/(cvU[0]-cvD[0]))*(bD[0]-bU[0]))-bD[0])
    ponto_b[1] = -((((cvD[1]-ponto_cv[1])/(cvD[1]-cvU[1]))*(bD[1]-bU[1]))-bD[1])
    ##coordenadas tipo BASE
   
    return ponto_b[0], ponto_b[1]
def Identificar_objeto(c,crop_img_contour, img):
    #logica para processar os dados de cada contorno encontrado
    # id do objeto
    i=0
    tabela = []
    for cnt in c:

        (x,y),_ = cv2.minEnclosingCircle(cnt) #função que ajuda a descobrir o centro do objeto
        center = (int(x),int(y))
        cv2.putText(crop_img_contour, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 8, cv2.LINE_AA) #escreve o Id do objeto no centro
       
        box= []
        rect = cv2.minAreaRect(cnt) #acha os pontos do retangulo q vai delimitar o objeto
        box_float = cv2.boxPoints(rect)
        box = np.int0(box_float) #conversão para manipulação
        #modo como o openCV numera os vértices
        ###3###
        #2###4#
        ###1##
        
        cv2.drawContours(crop_img_contour,[box],0,(225,0,255),2) #desenhar o retangulo

        #distancias entres os pontos pra saber qual o melhor lado pra pegar
        # distancia 0 garra anti horaria
        # distancia 1 garra horaria
        distancia_0= math.sqrt((box[0][0]-box[1][0])**2 + (box[0][1]-box[1][1])**2)
        distancia_1= math.sqrt((box[0][0]-box[3][0])**2 + (box[0][1]-box[3][1])**2)
        #delimitador de tamanho max de face
        limite = 300
        if ((distancia_0 > limite) and (distancia_1 > limite)):
            cv2.putText(crop_img_contour, "IMPOSSIVEL", center, cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 8, cv2.LINE_AA)
            
        else:
            #escolher a face de menor tamanho, pq uma delas pode ser fora do limite permitido (maior que a garra)
            #lógica para converter ângulo em ãngulo_braco (a logica muda de acordo com o sentido de giro)... o braço entende de -90 a 90
            if (distancia_0 > distancia_1):
                
                angulo = Angulo(box[0], box[3], center, img)
                angulo_braco = math.fabs(angulo) - 90
                angulo_braco = round(angulo_braco,2)

            else:
                angulo = Angulo(box[0], box[1], center, img)
                angulo_braco = 90 - angulo
                angulo_braco = round(angulo_braco,2)

           #para evitar angulos grandes e o braço girar mais do que deve. Essa logica inverte a face que o braço vai pegar
            if angulo_braco > 90:
                angulo_braco = angulo_braco - 180
            if angulo_braco < -90:
                angulo_braco = angulo_braco + 180
            #converter de pixls (x e y) para dados reais (braço)
            centro_conv_X, centro_conv_Y = Conversao_centro(center)
            #correção necessário, pois o ponto de giro não é fixo... lógica desenvolvida por mapeamento e obtenção de padrão de erro
            if angulo_braco < 0:
                centro_conv_X = centro_conv_X + (0.1625 * angulo_braco)
                centro_conv_X = round(centro_conv_X,3)
            
            #organizar para alimentar o dataframe
            objeto = []
            objeto.append(i) 
            objeto.append(centro_conv_X)   
            objeto.append(centro_conv_Y) 
            objeto.append(angulo_braco)   
            tabela.append(objeto)
        i += 1
    return crop_img_contour, tabela

def Camera():
    #utilizando a câmera
    captura = cv2.VideoCapture(0) #instancia o uso da webcam
 
    while(1):
        _, frame = captura.read() #pega efetivamente a imagem da webcam
           
        cv2.imshow('camera fixada? Aperte Esc para concluir',frame) #teste de angulo da câmer
    
        k = cv2.waitKey(10) & 0xff #fechar com Esc
        if k == 27:
            break
    
    captura.release() #dispensa o uso da webcam
    cv2.destroyAllWindows()
    return frame

def Varredura():
    #realiza a varredura e chama a função de analise de contornos
    img = Camera() 
    #caso queira desabilitar a camera, é so comentar a linha de cima e usar as de baixo 
    #img = cv2.imread("<endereço da imagem>")
    #img = cv2.resize(img, (639,479), interpolation = cv2.INTER_AREA) 
    #chama a função de enquadramento
    img = Enquadramento(img)
    
    #a função da janela declarada é impedir q a imagem apareça gigantesca e ñ perca qualidade
    cv2.namedWindow('Deseja Recortar a imagem?',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Deseja Recortar a imagem?',(int(479*proporcao),479))
    
   

    ##### rotacionar 180º para melhor compreenção
    altura, largura = img.shape[:2]
    ponto = (largura / 2, altura / 2) #ponto no centro da figura, ponto de giro
    rotacao = cv2.getRotationMatrix2D(ponto, 180, 1.0)
    img = cv2.warpAffine(img, rotacao, (largura, altura))
    
    ###primeira janela pra corte 
    print ("\n\n\n")
    print ("--Deseja recortar a imagem?--")
    print ("-Sim:\n  Selecione clicando e arrastando.\n  Confirma com a tecla Enter")
    print ("-Não:\n  Aperte a tecla Esc\n\n\n")
    xs,ys,w,h = cv2.selectROI('Deseja Recortar a imagem?',img) #permite o usuario cortar a imagem com o mouse
    cv2.destroyAllWindows()
    crop_img_true=crop_img_contour=img[ys:ys+h, xs:xs+w] #salvandoos dados do corte
    cortada = True
    if crop_img_true.shape[0]< 1:
        cortada = False
        crop_img_true=crop_img_contour=img
    print ("\n\n\n")
    print ("--Utilize as barras para evidenciar, da melhor forma possível, os objetos desejados--")
    print ("--O filtro dinâmico possui limitações, logo, talvez não seja possível evidenciar todos os objetos desejados de um só vez--")
    #filtro dinamico 
    desejo = "b"
    while 1:
        x,y,z,a,b,c=(tr.tracker(crop_img_true)) #biblioteca que gera filtro de cores de modo iterativo
        
        crop_img_true=cv2.cvtColor(crop_img_true,cv2.COLOR_BGR2HSV) #é necessario converter para hsv para passar o filtro

        #criando imagem base pra varredura com o contorno do filtro de cores 
        mask_inRange=cv2.inRange(crop_img_true,(x,y,z),(a,b,c))
        
        #lógica para sobreposição de filtros, necessário caso o usuário queira evidencia cores distantes no espectro HSVe com intervalos
        if desejo == "a":
            mask_inRange = cv2.resize(mask_inRange, (int(479*proporcao),479), interpolation = cv2.INTER_AREA)
            mask_inRange_temp = cv2.resize(mask_inRange_temp, (int(479*proporcao),479), interpolation = cv2.INTER_AREA)
            
            for y in range(0, mask_inRange.shape[0]): #percorre linhas
                for x in range(0, mask_inRange.shape[1]): #percorre colunas
                    #mask eh uma imagem composta apenas de 0 e 255
                    if (mask_inRange_temp[y][x] == (255)):
                        mask_inRange[y][x] = (255)
            mask_inRange = cv2.resize(mask_inRange, (int(2500*proporcao),2500), interpolation = cv2.INTER_AREA)

        print ("--Deseja avidenciar mais objetos?--")
        print (" a - Sim \n b - Não")
        desejo = input()
        if desejo == "a":
            mask_inRange_temp = mask_inRange
            crop_img_true=cv2.cvtColor(crop_img_true,cv2.COLOR_HSV2BGR)
        #caso o usuário envio qualquer coisa diferente de a, será considerado b   
        else:
            break


    #caso o usuário opte por recoertar a imagem, é necessario corrigir as medidas
    if cortada == True:

        blank_space_black= np.zeros((img.shape[0],img.shape[1]),np.uint8)
        blank_space_black[:]=(0)
        blank_space_black[ys:ys+h, xs:xs+w]= mask_inRange
        mask_inRange = blank_space_black
        crop_img_true=crop_img_contour=img
        
    #etapa importante para que a função findContours funcione sem problemas
    _, threshold = cv2.threshold(mask_inRange, 250, 255, cv2.THRESH_BINARY)

    #varredura de contornos
    contours=[]
    contours,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
   
    #listas de bordas 
    contornos_reais=[]
    total_objetos=0
    # tirando objetos menores que o minimo... retira ruido
    for cnt in contours:
        if cv2.contourArea(cnt)>=maxi:
            contornos_reais.append(cnt)
            total_objetos += 1

    print ("\n\n")
    print("-- Total de objetos encontrados",total_objetos," --") 
    # todos os dados são obtidos atraves da mascara, uma imagem praticamente binária.
    #as edições (contornos, escrever o ID no centro do objeto) são feitas em cima da imagem crop_img_contour que é a mesma imagem justificada, enquadrada lá do inicio
    #desenhar contornos
    cv2.drawContours(crop_img_contour, contornos_reais, -1, (0, 255, 0), 5)

    crop_img_contour, tabela = Identificar_objeto(contornos_reais, crop_img_contour, img) #chamar a função Identificar_objetos
    img_final= cv2.resize(crop_img_contour, (850,512), interpolation = cv2.INTER_AREA)
    cv2.imwrite("img_final.jpg", img_final) #salvando a imagem na pasta do programa
    
    ###tabela
    tabela_df = pd.DataFrame(data=tabela, columns = ['Id', 'Centro_X', 'Centro_Y','Ângulo',])
    tabela_df.to_csv("tabela.csv")

    
    return tabela_df, img_final

def Menu_varredura():
    #precisei criar como função para ter liberdade de chamar em loop sem problemas
    print("\n\n")
    print("###################### MENU ######################")
    print("###################-Varredura-####################")
    print(" a - Novo\n b - Atual")
    escolha = input()
    if (escolha == "a"):
        print ("-- Nova varredura selecionado --")
        tabela, img_final = Varredura()
    elif(escolha == "b"):
        #só haverá histórico caso ao menos uma varredura já tenha acontecido. 
        print ("-- Varredura atual selecionado --")
        tabela = pd.read_csv("tabela.csv")
        img_final = cv2.imread("img_final.jpg")
    else :
        print ("-- Selecione uma das opções --")
        tabela = Menu_varredura()
    print("\n\n\n")
    return tabela, img_final, escolha

def Enquadramento (img):
    #mesma logica do programa para seleção dos pontos para calibração e ajustes
    img = cv2.resize(img, (639,479), interpolation = cv2.INTER_AREA)
    pts1 = np.float32([pontos[0], pontos[1], pontos[3], pontos[2]])
    ##pts2 deve seguir a propporção do retangulo base densenhado pelo braço
    ## o retangulo que dita a proporção da workspace, importante delimita-la com o desenho progemado em .cod no braço (exem programa joe_ret2, passivo de alterações)
    pts2 = np.float32([[0,0], [int(2500*proporcao),0], [0,2500], [int(2500*proporcao),2500]])

    #ajusta a pespectiva deformando a imagem de modo a diminuir as distoções causadas pelo angulo da câmera
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(int(2500*proporcao),2500))

    return dst

def Envia(x,y,a):
    #socket com o CLP
    socket_i=0
    data=bytearray([0,0,0,0,0,0,0,0,0,0])
    Neg=0
    test=1 #flag para controlar escrita e leitura
    #estabelece comunicação
    if(socket_i ==0):
        serverHost = '192.168.1.200'
        serverPort = 2000
        sockobj = socket(AF_INET, SOCK_STREAM)
        sockobj.connect((serverHost, serverPort))

    neg=0
    tes=test.to_bytes(1,byteorder="big")
    X=x.to_bytes(4,byteorder="big")
    Y=y.to_bytes(4,byteorder="big")
    #lógica para trabalhar com negativos. Conversão no PDL
    if a<0:
        a=a*(-1)
        neg=1
    Neg= neg.to_bytes(1, byteorder="big")
    A=a.to_bytes(1,byteorder="big")
    
    #pacote mensagem
    data=X+Y+tes+Neg+A
    sockobj.send(data)
    time.sleep(3/100)
    
    test=0
    tes=test.to_bytes(1,byteorder="big")
    data=X+Y+tes+Neg+A
    sockobj.send(data)
    test=1
    print("###################################################################")
    print("                             ATENÇÂO")
    print("esteja proximo ao botão de emergencia, caso ocorra algum imprevisto\n")
    print("\n#################################################################")
    print("-- Em execução --")
    recebido_bytes = sockobj.recv(1)#espera a resposta do braço para dar continuidade
          
    socket_i=1


while 1:

    tabela, img_final, escolha= Menu_varredura()
    tabela = tabela.set_index("Id")
    
    if escolha == 'b': #usando o csv, vem com uma coluna index indesejada,
        tabela = tabela.drop(columns = ['Unnamed: 0']) #apagando a coluna index
    
    cv2.imshow("Objetos possiveis", img_final)
    print ("--Caso os objetos apresentados não seja os desejados, realize uma nova varredura--")
    k = cv2.waitKey(0) & 0xff #fechar com Esc
    if k == 27:
        cv2.destroyAllWindows()


    print("\n\n\n\n\n")
    while 1: 
        
        print("###################### MENU ######################")
        print("##########-Quadro de Objetos possiveis-###########\n\n",tabela,"\n\n") 
        print("-- Selecione o objeto desejado --")
        escolha = input()

        if int(escolha) in tabela.index:
            objeto_escolhido = tabela.loc[int(escolha)]
            objeto_escolhido_lista = objeto_escolhido.values

            print ("\n-- Objeto ",escolha, " selecionado --\n")
            print("\n\n\n")
            objeto_escolhido_lista2= objeto_escolhido_lista[-3:]    
            X= objeto_escolhido_lista2[0]
            X= round(X,0)
            X= int(X)
            Y= objeto_escolhido_lista2[1]
            Y= round(Y,0)
            Y= int(Y)
            A= objeto_escolhido_lista2[2]
            A= round(A,0)
            A= int(A)
            Envia (X, Y, A) #função envia pelo socket
            tabela = tabela.drop(int(escolha)) #atualiza a tabela de opções.. o arquivo csv continua intácto
        else:
            print("\n\n\n\n\n")
            print("-- Selecione um dos objetos disponiveis na tabela --")
        vazio = tabela.empty #teste para saber se ainda tem objetos na lista
        if vazio:
            print("-- Realize uma nova varredura --")
            break
            



