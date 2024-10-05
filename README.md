# layer-unlearning

1. Clone the repo
    ```
    git clone https://github.com/liuup/layer-unlearning
    ```
3. Params description  
   - `--model` or `-m`: choose the model: `resnet18` or `cnn`
   - `--batch` or `-b`: batch size
   - `--lr` or `-lr`: learning rate
   - `--round` or `-r`: overall round
   - `--epoch` or `-e`: epochs for every round
   - `--unlearn_epoch` or `-ue`: epochs for unlearning process
   - `--unlearn_k` or `-uk`: unlearning model layers
   - `--poison` or `-p`: poison data ratio

4. Run the code in the local
    ```
    cd layer-unlerarning

    python main.py -m cnn -b 128 -lr 0.001 -r 3 -e 16 -ue 10 -uk 3 -p 0.08
    ```