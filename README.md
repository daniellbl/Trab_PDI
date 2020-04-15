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

1._--img1_: Caminho para a primeira imagem;
2. _--img2_: Caminho para a segunda imagem;
3. _--saida_: Caminho onde será salvo a saída o programa;
4. _--extensao_: Formato do arquivo de saída.
5. _@--comando_: Qual a operação que deve ser feita, as opções são:

5.1 _gray_: Transforma a primeira imagem em escala de cinza;

5.2 _dif_: Calcula e salva imagem diferença entre duas imagens, também calcula o MSE e PSNR;

5.3 _hist_: Calcula o histograma da imagem (RGB ou cinza);

5.4 _rgb_: Salva cada canal de cor em uma imagem diferente;

5.5 _trans_log_: Transformação logarítmica;

5.6 _trans\_pow_: Transformação de potência

5.7 _hist\_eq_: Equalização de histograma

5.8 _binary_: Binarização da imagem

5.9 _fpb_: Filotrs passa baixa

5.10 _fpa_: Filotrs passa alta

Exemplo de chamada do programa:
```
python codigo.py --img1 a.png --img2 b.png -o out.png -c dif
```
