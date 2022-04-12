import numpy
import cv2

imagem = cv2.imread('coins.png')
cv2.imshow("Imagem", imagem)
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# uma estimativa aproximada das moedas. usando a binarização de Otsu
ret, thresh = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#  remove quaisquer pequenos ruídos brancos na imagem
kernel = numpy.ones((3, 3), numpy.uint8)
# para remover qualquer pequenos orifícios no objetos
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

# Assegura a área de fundo
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Encontrar uma área de primeiro plano segura
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Encontrando região desconhecida
sure_fg = numpy.uint8(sure_fg)
icog = cv2.subtract(sure_bg, sure_fg)

# Rotulagem do marcador
ret, marcador = cv2.connectedComponents(sure_fg)

# Adicione um a todos os rótulos para que o plano de fundo não seja 0, mas 1
marcador = marcador + 1

# marqua a região de incógnita com 0
marcador[icog == 255] = 0

# aplica watershed
marcador = cv2.watershed(imagem, marcador)
imagem[marcador == -1] = [255, 0, 0]


cv2.imshow("Imagem watershed", imagem)
cv2.waitKey(0)