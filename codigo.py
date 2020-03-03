import click
import numpy as np	
import math
import cv2 as cv
import io
import matplotlib.pyplot as plt 


PIXEL_MAX = 255.0
WAIT_TIME = 3

def save_format(img, name, format):
	outfile = name + "." + format
	cv.imwrite(outfile, img)

def get_img_from_fig(fig, dpi=90):
	buf = io.BytesIO()
	fig.savefig(buf, format="png", dpi=dpi)
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
	
def command_rgb(img):
	return cv.split(img)

@click.command()
@click.option("--img1", required=True, help="Path of the first image input file.", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--img2", help="Path of the second image input file.", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--output", "-o", "out_file", required=True, help="Path of the output file.", type=click.Path(exists=False, dir_okay=True, readable=False))
@click.option('--format', "-f", default="png", help='file format.')
@click.option('--command', "-c", required=True)
def process(img1, img2, out_file, format, command):
	img1 = cv.imread(img1, cv.IMREAD_UNCHANGED)
	show_img(img1, "Imagem 1")
	if img2 is not None:
		img2 = cv.imread(img2, cv.IMREAD_UNCHANGED)
		show_img(img2, "Imagem 2")
		save_format(img2, out_file, format)
	
	if command == "gray":
		img_resul = command_gray(img1)
		show_img(img_resul, "Imagem Resultado")
		save_format(img_resul, out_file, format)
	elif command == "dif":
		img_resul = command_dif(img1, img2)
		show_img(img_resul, "Imagem Resultado")
		save_format(img_resul, out_file, format)
		print("Erro médio quadrático = " + str(mean_square_error(img1, img2)))
		print("PSNR = " + str(psnr(img1, img2)))
	elif command == "hist":
		img_resul = command_hist(img1)
		show_img(img_resul, "Imagem Resultado")
		save_format(img_resul, out_file, format)
	elif command == "rgb":
		if len(img1.shape) == 3:
			channels = command_rgb(img1)
			show_img(channels[2], "Imagem Resultado - Vermelho")
			save_format(channels[2], out_file+"_r", format)
			show_img(channels[1], "Imagem Resultado - Verde")
			save_format(channels[1], out_file+"_g", format)
			show_img(channels[0], "Imagem Resultado - Azul")
			save_format(channels[0], out_file+"_b", format)

if __name__ == "__main__":
	process()