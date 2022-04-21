from sentence_transformers import SentenceTransformer, util

if __name__ == '__main__':
    # model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')

    query_embedding = model.encode('Network architectures or network communication protocols for network security for '
                                   'supporting authentication of entities communicating through a packet data network '
                                   'using an additional device, e.g. smartcard, SIM or a different communication '
                                   'terminal.')

    passage_embedding = model.encode(['Network architectures or network communication protocols for network'
                                      ' security for supporting authentication of entities communicating through '
                                      'a packet data network',
                                      'Network architectures or network communication protocols for network security for '
                                      'supporting authentication of entities communicating through a packet data network '
                                      'using an additional device, e.g. smartcard, SIM or a different communication '
                                      'terminal.'])

    print('Encod ', query_embedding)
    print("Similarity:", util.cos_sim(query_embedding, passage_embedding))