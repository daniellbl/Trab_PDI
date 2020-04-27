# Trab_PDI
Aluno: Daniel Lucas Braga Lucindo
Repositório do Trabalho de PDI.

# Instalação
1. Instalar Anaconda:
  Anaconda é um gerenciador de pacotes e um gerenciador de ambientes para _Python_ e _R_.
[Link](https://www.anaconda.com/distribution/)
2. Abrir o terminal do Anaconda.
3. Executar os seguintes códigos
```python
  conda install -c conda-forge opencv
  pip install click
  pip install matplotlib
  pip install numpy
```

# Execução
Para executar o programa basta passar os argumentos necessários pela linha de comando, os possíveis argumentos são

1. _--img1_: Caminho para a primeira imagem;

1. _--img2_: Caminho para a segunda imagem;

1. _--saida_: Caminho onde será salvo a saída o programa;

1. _--extensao_: Formato do arquivo de saída, exemplo: "png", "jpg".

1. _@--comando_: Qual a operação que deve ser feita, as opções são:

   1. _gray_: Transforma a primeira imagem em escala de cinza;

   1. _dif_: Calcula e salva imagem diferença entre duas imagens, também calcula o MSE e PSNR;

   1. _hist_: Calcula o histograma da imagem (RGB ou cinza);

   1. _rgb_: Salva cada canal de cor em uma imagem diferente;

   1. _trans_log_: Transformação logarítmica;

   1. _trans\_pow_: Transformação de potência;

   1. _hist\_eq_: Equalização de histograma;

   1. _binary_: Binarização da imagem;

   1. _fpb_: Filtros passa baixa;

   1. _fpa_: Filtros passa alta.
   
   1. _erosao_: Erosão de imagens binárias.
   
   1. _dilat_: Dilatação de imagens binárias.

1. _@--gamma_: Parâmetro para a função de transformação de potência;

1. _@--limiar_: Parâmetro usado na função de binarização;

1. _@--kernel_: Formato do kernel usado, as opções são:
   
   1. _ret_: Retangular;

   1. _eli_: Elipse;
   
   1. _crz_: Em cruz.

1. _@--kernel\_x_: Tamanho do kernel em X;

1. _@--kernel\_y_: Tamanho do kernel em Y;

1. _@--kernel_: Formato do kernel usado, as opções são:

1. _@--desvio_: Desvio padrão usado no filtro Gaussiano;

1. _@--filtro_: Qual a filtro que deve ser feito, as opções são:

   1. _media_, _gaus_ e _mediana_ para filtros passa baixa.

   1. _lap_ e _sob_ para filtros passa alta.

Exemplo de chamada do programa:
```
python codigo.py --img1 a.png --img2 b.png --saida out.png --comando dif
```
