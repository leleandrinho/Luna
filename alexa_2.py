# Importando bibliotecas

import speech_recognition as sr
import pyttsx3
import requests
import json
import datetime
import asyncio
from shazamio import Shazam
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np
from deep_translator import GoogleTranslator
import easyocr

#chamando url api llama e inicializando voz Luna
url = "http://localhost:11434/api/generate"
luna = pyttsx3.init()

#ajustando voz Luna
luna.setProperty('voice', r'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_PT-BR_MARIA_11.0')
luna.setProperty('volume', 1.0)
luna.setProperty('rate', 160)

# iniciando reconhecedor de voz
reconhecedor = sr.Recognizer()


# iniciando modelo reconhecimento facial
detectorFace = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
reconhecedor_face = cv2.face.EigenFaceRecognizer_create()
reconhecedor_face.read('classificadoreigen.yml')

# configurações do tradutor
tradutor = GoogleTranslator(source="en", target="pt")

# funcao para luna reconhecer a facial
def reconhecer_facial():

        # passando parametro do tamanho do reconhecimento facial
        largura, altura = 220, 220
        #fonte que irá escrever o nome em tela
        font = cv2.QT_FONT_NORMAL
        # vendo qual camera sera usada do sistema
        camera = cv2.VideoCapture(0)

        # variavel que iniciara/finalizara o processo
        fim = False

        # loop de reconhecimento
        while not fim:
            # iniciando camera
            status, imagem = camera.read()

            # tranformando imagem em escalas de cinza para um melhor reconhecimento (sem ser camada RGB)
            imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

            # detector de faces
            facesDetectadas = detectorFace.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(30, 30))

            # iniciando loop para reconhecer a face
            try:
                for x, y, l, a in facesDetectadas:
                    # pegando imagem da face (foto)
                    imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (altura, largura))
                    # criando parametros para imagem
                    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
                    # vendo quem é de acordo com o modelo de treinamento
                    nome, confianca = reconhecedor_face.predict(imagemFace)

                    # printando a confiança (etapa não necessária)
                    # print(confianca)

                    # caso a face seja conhecida, ele irá retornar o "ID", verificando "ID" com o nome da pessoa
                    # caso seja o leandro (ID 1), ele libera, senão, não
                    if nome == 1:
                        nome = "Leandro"
                        # retornando True para finalizar o "programa" de reconhecimento
                        fim = True
                        # retornando return para saber que foi reconhecido a imagem
                        return True, nome

                    # printando nome da pessoa reconhecida na tela
                    cv2.putText(imagem, str(nome), (x, y + altura - 20), font, 2, (0, 0, 255))
            except:
                # caso de errado, ele irá printar a seguinte imagem
                print("Não consegui reconhecer...")

            # inicando "programa" da camera para reconhecer face
            cv2.imshow("Faces", imagem)

            # caso o rosto não seja reconhecido, ou de erro, o usuario podera clicar em "Q" para finalizar o processo de reconhecimento
            if cv2.waitKey(1) == ord('q'):
                fim = True
                return False, ""


# criando classe e funções
# classe Luna recebendo a fala (comando)
class Luna():
    def __init__(self, fala):
        self.fala = fala

    # funçã para reconhecer objetos
    def reconhecer_objeto(self):
        fala = self.fala

        # iniciando camera
        camera = cv2.VideoCapture(0)

        # contador de amostra
        amostra = 0

        while True:
            status, imagem = camera.read()
            # quando clica na tecla Q ele tira a foto
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Média: ", np.average(imagem))

                # colocando a imagem dentro do diretório
                localFoto = 'foto_objeto/' + str("objeto") + '.jpg'
                cv2.imwrite(localFoto, imagem)
                amostra += 1

            cv2.imshow("objeto detectado: ", imagem)
            # quando o ocntador for igual a 1 (imagem) o programa para de rodar
            if amostra == 1:
                break

        print("Amostras capturadas com sucesso!")
        camera.release()
        cv2.destroyAllWindows()


        # iniciando reconhecimento dos objetos da imagem
        try:
            # traz a imagem que acabou de ser tirada
            imagem = cv2.imread("foto_objeto/objeto.jpg")
            # baixa o modelo e faz a detecção
            bbox, label, conf = cv.detect_common_objects(imagem, confidence=0.25, model='yolov3-tiny')
            # printa o resultado
            print(label)

            out = draw_bbox(imagem, bbox, label, conf, write_conf=True)

            cv2.imshow("Objeto detectado", out)
            cv2.waitKey()
        # caso de erro na detcção irá aparecer um aviso de erro
        except Exception as erro:
            print('Erro: ', erro)

        # Luna fala objetos detectados
        print("LUNA: Na sua imagem há: ")
        luna.say("Na sua imagem há")
        luna.runAndWait()

        # caso não tenha nada dentro da lista de resultado, ele irá retornar essa mensagem, senão irá para o ELSE
        if not label:
            print(f"LUNA: Não reconheci nenhum objeto")
            luna.say("Não reconheci nenhum objeto")
            luna.runAndWait()
        else:
            # para cada palavra dentro da lista ele fará a tradução e a Luna irá dizer
            for i in label:
                traducao = tradutor.translate(i)
                print(f"LUNA: {traducao}")
                luna.say(traducao)
                luna.runAndWait()


    # Função para reconhecimento de Texto (OCR)
    def reconhecer_texto(self):
        fala = self.fala

        # baixando modelo
        reader = easyocr.Reader(['pt', 'en'])

        # iniciando camera
        camera = cv2.VideoCapture(0)
        amostra = 0

        while True:
            status, imagem = camera.read()
            # Tira a foto quando aperta a tecla Q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Média: ", np.average(imagem))

                # salva a foto no diretório
                localFoto = 'foto_ocr/' + str("ocr") + '.jpg'
                cv2.imwrite(localFoto, imagem)
                amostra += 1

            cv2.imshow("Palavra detectada: ", imagem)
            if amostra == 1:
                break

        print("Amostras capturadas com sucesso!")
        camera.release()
        cv2.destroyAllWindows()

        # resultado da detecção de palavras (detail = 0, irá passar apenas o resultado, sem parametros)
        resultado = reader.readtext('foto_ocr/ocr.jpg', detail=0)

        # Luna diz os objetos
        print("LUNA: Na sua imagem está escrito: ")
        luna.say("Na sua imagem está escrito")
        luna.runAndWait()

        # Caso não tenha nada na lista do resultado, ela irá dar a mensagem, senão irá pro ELSE
        if not resultado:
            print(f"LUNA: Não reconheci nenhuma palavra")
            luna.say("Não reconheci nenhuma palavra")
            luna.runAndWait()
        else:
            # para cada texto/palavra detctada ela irá falar
            for i in resultado:
                print(f"LUNA: {i}")
                luna.say(i)
                luna.runAndWait()





    # funçao de chamada: irá acionar toda vez que chamarmos a Luna
    def chamada(self):
        fala = self.fala
        if fala.lower() in ['ok luna', 'okluna', 'ok lunas', 'ok, luna']:
            print("LUNA: Qual a sua pergunta? ")
            luna.say("Qual a sua pergunta?")
            luna.runAndWait()
            return True

        else:
            print("LUNA: Desculpa, não te escutei...")
            luna.say("Desculpa, não te escutei...")
            luna.runAndWait()
            return False



# função de pergunta, sempre que houver alguma pergunta sem a necessidade do uso de APIs, irá retornar a IA LLAMA 3.1
    def pergunta(self):
        fala = self.fala
        print("LUNA: Aguarde...")
        luna.say("Aguarde...")
        luna.runAndWait()
        # prompt llama
        input_json = {
            "model": "llama3.1",
            "prompt": "responda sucitamente em português em poucas palavras em um parágrafo" + fala
        }
        #chamando resposta llama
        response = requests.post(url, json=input_json)
        #fazendo tratamento da resposta llama
        linhas = response.text.strip().split('\n')
        valores_response = []
        for linha in linhas:
            # carregar a linha como um dicionario python
            obj = json.loads(linha)
            # obter o valor da chave 'response'
            resposta = obj.get('response')
            # adicionar a lista de valores de 'response'
            valores_response.append(resposta)
        # juntar os valores de 'response' em uma unica string
        nova_string = ''.join(valores_response)
        # exibir e falando a nova string resultante
        print("LUNA: " + nova_string)
        luna.say(str(nova_string))
        luna.runAndWait()


# função para marcar novo compromisso na agenda.txt
    def marcar_compromisso(self):
        fala = self.fala
        print("LUNA: Qual evento deseja marcar?")
        luna.say("Qual evento deseja marcar?")
        luna.runAndWait()
        audio = reconhecedor.listen(mic)
        texto = reconhecedor.recognize_google(audio, language='pt')
        print("VOCÊ: " + texto)
        # abrindo agenda.txt e adicionando o texto do comando (compromisso)
        agenda = open("agenda.txt", "a+", encoding='UTF8')
        agenda.write(f"\n{texto}")
        print("LUNA: ", texto, "marcado na agenda")
        luna.say(f"{texto} marcado na agenda")
        luna.runAndWait()


# função para ver compromissos da agenda.txt
    def ver_compromissos(self):
        fala = self.fala
        # lendo agenda.txt
        agenda = open("agenda.txt", "r", encoding='UTF8')
        print("LUNA: Seus próximos eventos são")
        luna.say("Seus próximos eventos são")
        luna.runAndWait()
        # Luna listando todos as linhas (compromissos) da agenda.txt
        for linha in agenda:
            print("LUNA: ", linha)
            luna.say(linha)
            luna.runAndWait()


# função que retorna dia e hora atual
    def hora_data_atual(self):
        fala = self.fala
        # variaveis de dia, hora, minuto, etc...
        agora = datetime.datetime.now()
        horas = agora.hour
        minutos = agora.minute
        dia = agora.today().day
        mes = agora.today().month
        ano = agora.today().year
        # dicionário com numero do mes e nome do mes para melhor resposta
        meses = {
            1: "Janeiro",
            2: "Fevereiro",
            3: "Março",
            4: "Abril",
            5: "Maio",
            6: "Junho",
            7: "Julho",
            8: "Agosto",
            9: "Setembro",
            10: "Outubro",
            11: "Novembro",
            12: "Dezembro"
        }
        # reconhecendo fala para Luna dizer horário ou dia atual
        if fala.lower() in ['que horas sao', 'que horas são', 'qual é o horário agora', 'qual e o horario agora', 'que horas são agora', 'que horas sao agora']:
            print("LUNA: Agora são",horas, "horas e", minutos, "minutos")
            luna.say(f"agora são {horas} horas e {minutos} minutos")
            luna.runAndWait()
        elif fala.lower() in ['que dia é hoje', 'que dia e hoje']:
            print(f"LUNA: Hoje é dia {dia} de {meses[mes]} de {ano}")
            luna.say(f"Hoje é dia {dia} de {meses[mes]} de {ano}")
            luna.runAndWait()



# função para clima atual na região X
    def clima_atual(self):
        fala = self.fala
        # chave API
        API_KEY = "12e89494e8d6d12e0b02689a27e65814"
        print("LUNA: De qual cidade você quer saber?")
        luna.say("De qual cidade você quer saber?")
        luna.runAndWait()
        audio = reconhecedor.listen(mic)
        texto = reconhecedor.recognize_google(audio, language='pt')
        print("VOCÊ: " + texto)
        # reconhecendo a cidade a partir do comando
        cidade = texto
        # linkando com a API
        link = f"https://api.openweathermap.org/data/2.5/weather?q={cidade}&appid={API_KEY}&lang=pt_br"
        # fazendo a requisição, Luna printando e falando
        requisicao = requests.get(link)
        requisicao_dic = requisicao.json()
        descricao = requisicao_dic['weather'][0]['description']
        temperatura = requisicao_dic['main']['temp'] - 273.15
        print(f"LUNA: O clima agora em {cidade} é {descricao}, fazendo {temperatura:.2f}ºC")
        luna.say(f"O clima agora em {cidade} é {descricao}, fazendo {temperatura:.2f}ºC")
        luna.runAndWait()


# função para encontrar musica com shazam
    async def encontar_musica(self):
        shazam = Shazam()
        # buscando musica pelo arquivo musica.wav (arquivo contendo audio do microfone com a musica desejada)
        out = await shazam.recognize('musica.wav')
        # Luna printando e falando nome da musica e artista principal
        print(f"LUNA: Essa musica se chama {out['track']['title']}, do artista {out['track']['subtitle']}")
        luna.say(f"Essa musica se chama {out['track']['title']}, do artista {out['track']['subtitle']}")
        luna.runAndWait()


# função para ver cotação das principais moedas: dólar, euro e bitcoin
# essa função recebe também a moeda especificada no comando
    def cotacao(self, moeda):
        # fazendo link com API
        url = "https://economia.awesomeapi.com.br/last/USD-BRL,EUR-BRL,BTC-BRL"
        # obtendo a requisição
        response = requests.get(url)
        # Luna printando e falando nome da moeda e cotação atual
        # a API atualiza de 30 em 30 segundos, valores sempre atualizados
        print(f"LUNA: No momento ele está custando {int(float(response.json()[moeda]['bid']))} reais e {int(round(float(response.json()[moeda]['bid']) - int(float(response.json()[moeda]['bid'])),2) * 100)} centavos")
        luna.say(f"No momento ele está custando {int(float(response.json()[moeda]['bid']))} reais e {int(round(float(response.json()[moeda]['bid']) - int(float(response.json()[moeda]['bid'])),2) * 100)} centavos")
        luna.runAndWait()




# iniciando reconhecimento (ele passa o parametro true caso tenha liberado o reconhecimento, e o nome do usuário
facial,nome = reconhecer_facial()


# caso o reconhecimento seja liberado (TRUE)
if facial:
    try:
        # prompt dos comandos
        # configurando o microfone da maquina
        with sr.Microphone() as mic:
            # ajustando barulhos de fundo
            reconhecedor.adjust_for_ambient_noise(mic, duration=2)
            # abertura Luna
            print("""      
            ██╗░░░░░██╗░░░██╗███╗░░██╗░█████╗░
            ██║░░░░░██║░░░██║████╗░██║██╔══██╗
            ██║░░░░░██║░░░██║██╔██╗██║███████║
            ██║░░░░░██║░░░██║██║╚████║██╔══██║
            ███████╗╚██████╔╝██║░╚███║██║░░██║
            ╚══════╝░╚═════╝░╚═╝░░╚══╝╚═╝░░╚═╝      """)
            print("Usuário:",nome)
            print(f"LUNA: Olá {nome}!")
            luna.say("Olá" + nome)
            luna.runAndWait()


            # looping para continuar as perguntas até o silencio
            while True:
                # a variavel chamou serve para continuar toda vez que o OK luna for dito.
                chamou = True
                print("LUNA Escutando...")
                # ouvindo voz e transcrevendo
                audio = reconhecedor.listen(mic)
                texto = reconhecedor.recognize_google(audio, language='pt')
                print("VOCÊ: " + texto)
                # chamando a função padrão de chamada
                chamou = Luna(texto).chamada()
                # caso tenha dado erro ou não tenha entendido ele irá aparecer a seguinte mensagem
                if not chamou:
                    print("ERRO!")
                else:
                    # após chamada confirmada, real comando sera gravado e transcrito
                    audio = reconhecedor.listen(mic)
                    texto = reconhecedor.recognize_google(audio, language='pt')
                    print("VOCÊ: " + texto)
                    # inciando condições para chamada de funções
                    # cada IF, ELIF possui várias frases (formas de chamadas) em listas para cada função, caso essas frases estejam no texto transcrito, respectiva função sera chamada
                    # todos os IF, ELIF e ELSE tem seus respectivos textos (comando) transformados em lower (minusculo)
                    if texto.lower() in ['marcar compromisso na agenda', 'cadastrar evento na agenda',
                                         'cadastrar compromisso na agenda', 'cadastrar novo evento na agenda',
                                         'marcar evento na agenda', "marcar novo evento na agenda"]:
                        Luna(texto).marcar_compromisso()
                    elif texto.lower() in ['ler agenda', 'ler compromissos', 'meus compromissos', 'ver compromissos']:
                        Luna(texto).ver_compromissos()
                    elif texto.lower() in ['que horas sao', 'que horas são', 'qual é o horário agora',
                                           'qual e o horario agora', 'que horas são agora', 'que horas sao agora',
                                           'que dia é hoje', 'que dia e hoje']:
                        Luna(texto).hora_data_atual()
                    elif texto.lower() in ['qual o tempo agora', 'quantos graus esta agora', 'quantos graus está agora',
                                           "qual a temperatura atual", 'quantos graus está fazendo',
                                           'quantos graus está fazendo agora', 'graus está fazendo']:
                        Luna(texto).clima_atual()
                    # chamada da função shazam
                    elif texto.lower() in ['encontre essa música', 'que música é essa', 'shazam', 'encontrar música']:
                        # quando tais frases são identificadas no texto de comando, a Luna irá responder e "ouvir" o áudio da musica
                        reconhecedor.adjust_for_ambient_noise(mic, duration=2)
                        print("LUNA: Ouvindo música...")
                        luna.say("Ouvindo música...")
                        luna.runAndWait()
                        # ouvindo audio da musica pelo microfone
                        audio = reconhecedor.listen(mic)
                        # após terminar de tocar o trecho da musica, o arquivo musica.wav será sobrescrito com o áudio do microfone
                        # de preferncia o audio deve ser limpo e relativamente alto, sons muito baixos (speaker do celular por exemplo) dificilente são identificadas
                        with open("musica.wav", "wb") as f:
                            f.write(audio.get_wav_data())
                        print("LUNA: Buscando música...")
                        luna.say("Buscando música...")
                        luna.runAndWait()
                        # criando loop de busca
                        loop = asyncio.get_event_loop()
                        # quando encontrado, ele chama a função de encontrar musica com o arquivo ja sobrescrito
                        loop.run_until_complete(Luna(texto).encontar_musica())
                    elif texto.lower() in ["qual o valor do dólar", "qual é o valor do dólar", "quanto está o dólar",
                                           "dólar hoje", "valor do dólar", "valor dólar"]:
                        # função para valor do dólar recebe a "chave" de reconhcimento da moeda para API. Mesma lógica para as outras
                        Luna(texto).cotacao("USDBRL")
                    elif texto.lower() in ["qual o valor do euro", "qual é o valor do euro", "quanto está o euro",
                                           "euro hoje", "valor do euro", "valor euro"]:
                        Luna(texto).cotacao("EURBRL")
                    elif texto.lower() in ["qual o valor do bitcoin", "qual é o valor do bitcoin",
                                           "quanto está o bitcoin", "bitcoin hoje", "valor do bitcoin",
                                           "valor bitcoin"]:
                        Luna(texto).cotacao("BTCBRL")
                    elif texto.lower() in ['reconheça meus objetos', 'o que tenho aqui', 'quais são meus objetos', 'reconhecer objetos', 'reconhecer objeto', 'reconheça meu objeto']:
                        print("LUNA: Clique na tecla Q para tirar uma foto")
                        luna.say("Clique na tecla Q para tirar uma foto")
                        luna.runAndWait()
                        Luna(texto).reconhecer_objeto()
                    elif texto.lower() in ['o que esta escrito aqui', 'reconhecer texto', 'o que está escrito aqui', 'leia esse texto']:
                        print("LUNA: Clique na tecla Q para tirar uma foto")
                        luna.say("Clique na tecla Q para tirar uma foto")
                        luna.runAndWait()
                        Luna(texto).reconhecer_texto()
                    # caso nenhum dos IF, ELIF sejam chamados ou ativado, a Luna irá retornar com alguma resposta criativa vindo da IA llama 3.1
                    else:
                        Luna(texto).pergunta()

    # caso o programa "quebre" por conta do silencio, a luna irá entender que não existe mais comando, então dará tchau
    except sr.UnknownValueError:
        print("LUNA: Até Mais!")
        luna.say("Até Mais")
        luna.runAndWait()

# caso o acesso do reconhecimento facial não funcione ou não reconheça, ele irá aparecer a seguinte mensagem
else:
    print("Acesso Negado!")
    luna.say("Acesso Negado!")
    luna.runAndWait()