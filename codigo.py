import click
import numpy as np	
import math
import cv2 as cv
import io
import matplotlib.pyplot as plt 


PIXEL_MAX = 255.0
EXTENSAO_DEFAULT = "png"
LIMIAR_DEFAULT = 127
KERNEL_DEFAULT = "ret"
KERNEL_SIZE_DEFAULT = 3
DESVIO_DEFAULT = 1
GAMMA_DEFAULT = 1
WAIT_TIME = 10000

def save_format(img, name, extensao = EXTENSAO_DEFAULT):
	outfile = name + "." + extensao
	cv.imwrite(outfile, img)

def get_img_from_fig(fig, dpi=90):
	buf = io.BytesIO()
	fig.savefig(buf, format=EXTENSAO_DEFAULT, dpi=dpi)
	buf.seek(0)
	img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
	buf.close()
	img = cv.imdecode(img_arr, 1)
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	return img

def show_img(img, imgName):
	cv.imshow(imgName, img)
	cv.waitKey(WAIT_TIME)
	cv.destroyAllWindows()

def mean_square_error(img1, img2):
	return np.mean( (img1 - img2) ** 2 )

def psnr(img1, img2):
	mse = mean_square_error(img1, img2)
	if mse == 0:
		return 100
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def make_kernel(kernel, kernel_x, kernel_y):
	kernel_x = int(kernel_x)
	kernel_y = int(kernel_y)
	if kernel == "ret":
		return cv.getStructuringElement(cv.MORPH_RECT,(kernel_x, kernel_y))
	if kernel == "eli":
		return cv.getStructuringElement(cv.MORPH_ELLIPSE,(kernel_x, kernel_y))
	if kernel == "crz":
		return cv.getStructuringElement(cv.MORPH_CROSS,(kernel_x, kernel_y))

def command_gray(img):
	try:
		img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		return img_gray
	except:
		return img

def command_dif(img1, img2):
	return np.absolute(img1 - img2)

def command_hist(img):
	fig = plt.figure()
	if len(img.shape) == 3:
		color = ('b','g','r')
		for i,col in enumerate(color):
			histr = cv.calcHist([img],[i],None,[256],[0,256])
			plt.plot(histr,color = col)
			plt.xlim([0,256])
	else:
		plt.hist(img.ravel(),256,[0,256])
	return get_img_from_fig(fig)
	
def command_log_transf(img):
	img_resul = np.zeros((img.shape[0],img.shape[1],1), np.float64)
	img = command_gray(img)
	for i in range(0, img.shape[0]):
		for j in range(0, img.shape[1]):
			img_resul[i, j] = math.log10(1 + img[i, j])
	const = 0
	max = 0
	for i in range(0, img_resul.shape[0]):
		for j in range(0, img_resul.shape[1]):
			if max < img_resul[i, j]:
				max = img_resul[i, j]
	const = PIXEL_MAX / max
	for i in range(0, img_resul.shape[0]):
		for j in range(0, img_resul.shape[1]):
			img_resul[i, j] = const * img_resul[i, j]
	print("const:" + str(const))
	return img_resul.astype('uint8')

def command_pow_transf(img, gamma = GAMMA_DEFAULT):
	gamma = np.array(gamma, dtype = np.float64)
	img_resul = np.zeros((img.shape[0],img.shape[1],1), np.float64)
	img = command_gray(img)
	for i in range(0, img.shape[0]):
		for j in range(0, img.shape[1]):
			img_resul[i, j] = np.power(img[i, j], gamma)
	const = 0
	max = 0
	for i in range(0, img_resul.shape[0]):
		for j in range(0, img_resul.shape[1]):
			if max < img_resul[i, j]:
				max = img_resul[i, j]
	const = PIXEL_MAX / max
	for i in range(0, img_resul.shape[0]):
		for j in range(0, img_resul.shape[1]):
			img_resul[i, j] = const * img_resul[i, j]
	print("const:" + str(const))
	return img_resul.astype('uint8')

def command_binary(img, treshold = LIMIAR_DEFAULT):
	img = command_gray(img)
	ret,img_resul = cv.threshold(img, int(treshold), 255, cv.THRESH_BINARY)
	return img_resul

def command_hist_eq(img):
	img = command_gray(img)
	img_resul = cv.equalizeHist(img)
	return img_resul

def command_FPB_avg(img, kernel_x = KERNEL_SIZE_DEFAULT, kernel_y = KERNEL_SIZE_DEFAULT):
	kernel_size = int(kernel_size)
	img_resul = cv.blur(img,(kernel_size,kernel_size))
	return img_resul

def command_FPB_gaus(img, kernel_x = KERNEL_SIZE_DEFAULT, kernel_y = KERNEL_SIZE_DEFAULT, desv = DESVIO_DEFAULT):
	kernel_size = int(kernel_size)
	desv = int(desv)
	img_resul = cv.GaussianBlur(img,(kernel_size,kernel_size), desv)
	return img_resul

def command_FPB_median(img, kernel_x = KERNEL_SIZE_DEFAULT, kernel_y = KERNEL_SIZE_DEFAULT):
	kernel_size = int(kernel_size)
	img_resul = cv.medianBlur(img, kernel_size)
	return img_resul

def command_FPA_laplace(img):
	img_resul = cv.Laplacian(img, cv.CV_8U)
	return img_resul

def command_FPA_sobel(img, kernel_x = KERNEL_SIZE_DEFAULT, kernel_y = KERNEL_SIZE_DEFAULT):
	kernel_size = int(kernel_size)
	img_resul = cv.Sobel(img,cv.CV_8U,1,0,ksize=kernel_size)
	show_img(img_resul, "Imagem Resultado")
	img_resul = cv.Sobel(img,cv.CV_8U,0,1,ksize=kernel_size)
	show_img(img_resul, "Imagem Resultado")
	img_resul = cv.Sobel(img,cv.CV_8U,1,1,ksize=kernel_size)
	show_img(img_resul, "Imagem Resultado")
	return img_resul

def command_erosao (img, kernel = KERNEL_DEFAULT, kernel_x = KERNEL_SIZE_DEFAULT, kernel_y = KERNEL_SIZE_DEFAULT):
	erosion = cv.erode(img, make_kernel(kernel, kernel_x, kernel_y), iterations = 1)
	return erosion

def command_dilatacao (img, kernel = KERNEL_DEFAULT, kernel_x = KERNEL_SIZE_DEFAULT, kernel_y = KERNEL_SIZE_DEFAULT):
	dilation = cv.dilate(img, make_kernel(kernel, kernel_x, kernel_y), iterations = 1)
	return dilation

def command_rgb(img):
	return cv.split(img)

@click.command()
@click.option("--img1", required=True, help="Caminho para a primeira imagem de entrada.", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--img2", help="Caminho para a segunda imagem de entrada.", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--saida", "out_file", required=True, help="Caminho para o arquivo de saída.", type=click.Path(exists=False, dir_okay=True, readable=False))
@click.option('--extensao', default="png", help='Extensão do arquivo.')
@click.option('--comando', required=True)
@click.option('--gamma')
@click.option('--limiar')
@click.option('--kernel')
@click.option('--kernel_x')
@click.option('--kernel_y')
@click.option('--desvio')
@click.option('--filtro')
def process(img1, img2, out_file, extensao, comando, gamma, limiar, kernel, kernel_x, kernel_y, desvio, filtro):
	img1 = cv.imread(img1, cv.IMREAD_UNCHANGED)
	show_img(img1, "Imagem 1")
	if img2 is not None:
		img2 = cv.imread(img2, cv.IMREAD_UNCHANGED)
		show_img(img2, "Imagem 2")
	if comando == "cinza":
		img_resul = command_gray(img1)
	elif comando == "dif":
		img_resul = command_dif(img1, img2)
		print("Erro médio quadrático = " + str(mean_square_error(img1, img2)))
		print("PSNR = " + str(psnr(img1, img2)))
	elif comando == "hist":
		img_resul = command_hist(img1)
	elif comando == "trans_log":
		img_resul = command_log_transf(img1)
	elif comando == "trans_pow":
		img_resul = command_pow_transf(img1, gamma)
	elif comando == "hist_eq":
		img_resul = command_hist_eq(img1)
	elif comando == "binary":
		img_resul = command_binary(img1, limiar)
	elif comando == "fpb":
		if filtro == "media":
			img_resul = command_FPB_avg(img1, kernel_x, kernel_y)
		elif filtro == "gaus":
			img_resul = command_FPB_gaus(img1, kernel_x, kernel_y, desvio)
		elif filtro == "mediana":
			img_resul = command_FPB_median(img1, kernel_x, kernel_y)
	elif comando == "fpa":
		if filtro == "lap":
			img_resul = command_FPA_laplace(img1)
		if filtro == "sob":
			img_resul = command_FPA_sobel(img1, kernel_x, kernel_y)
	elif comando == "erosao":
		img_resul = command_erosao(img1, kernel, kernel_x, kernel_y)
	elif comando == "dilatacao":
		img_resul = command_dilatacao(img1, kernel, kernel_x, kernel_y)
	elif comando == "rgb":
		if len(img1.shape) == 3:
			channels = command_rgb(img1)
			show_img(channels[2], "Imagem Resultado - Vermelho")
			save_format(channels[2], out_file+"_r", extensao)
			show_img(channels[1], "Imagem Resultado - Verde")
			save_format(channels[1], out_file+"_g", extensao)
			show_img(channels[0], "Imagem Resultado - Azul")
			save_format(channels[0], out_file+"_b", extensao)
			return
	
	show_img(img_resul, "Imagem Resultado")
	save_format(img_resul, out_file, extensao)

if __name__ == "__main__":
	process()