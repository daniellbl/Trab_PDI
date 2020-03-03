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

2._--img2_: Caminho para a segunda imagem;

3._--output_ ou _-o_: Caminho onde será salvo a saída o programa;

4._--format_ ou _-f_: Formato do arquivo de saída.

5._@--command_ ou _-c_: Qual a operação que deve ser feita, as opções são:
5.1_gray_(transforma a primeira imagem em escala de cinza);

5.2_dif_(Calcula e salva imagem diferença entre duas imagens, também calcula o MSE e PSNR);

5.3_hist_(Calcula o histograma da imagem (RGB ou cinza));

5.4_rgb_(Salva cada canal de cor em uma imagem diferente);

Exemplo de chamada do programa:
```
python asd.py --img1 a.png --img2 b.png -o out.png -c dif
```
