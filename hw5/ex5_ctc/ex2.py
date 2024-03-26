import sys
import numpy as np

EPS = 'eps'

def main(args):
    #load the numpy matrix from the file
    model_probs = np.load(args[1] if len(args) > 1 else 'idanush.npy')
    output = args[2] if len(args) > 1 else 'a'
    alphabet = args[3] if len(args) > 1 else 'ab'
    return ctc(model_probs, alphabet, output)

def calc_item(alpha, s, t, Z, model_probs, alphabet):
    Zs = Z[s]
    Zs_idx = alphabet.index(Zs) if Zs != EPS else -1
    if Z[s] == EPS or Z[s] == Z[s-2]:
        return (alpha[s-1, t-1] + alpha[s, t-1])*model_probs[t, Zs_idx]
    else:
        return (alpha[s-1, t-1] + alpha[s, t-1] + alpha[s-2, t-1])*model_probs[t, Zs_idx]
    
    
    
def ctc(model_probs, alphabet, transcription):
    #initialize the variables   
    Z = []
    for t in transcription:
        Z.append(EPS)
        Z.append(t)
    Z.append(EPS)

    #initialize the alpha and beta variables
    alpha = np.zeros((len(Z), model_probs.shape[0]))

    alpha[0][0] = model_probs[0, -1] # epsilon
    z1_idx = alphabet.index(transcription[0])
    alpha[1][0] = model_probs[0, z1_idx] # first character



    for t in range(1, model_probs.shape[0]):
        for s in range(0, len(Z)):
            alpha[s, t] = calc_item(alpha, s, t, Z, model_probs, alphabet)

    prob = 0
    for i in range(len(Z)):
        word = Z[:i+1]
        word = [w for w in word if w != EPS]
        if ''.join(word) == transcription:        
            prob += alpha[i, -1]
    return round(prob, 2)
    


if __name__ == '__main__':
    args = sys.argv
    print(main(args))

