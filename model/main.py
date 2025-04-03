def main ():
    model = create_model ()

    train (model)

    evaluate (model)

if __name__=='__main__':
    main() 