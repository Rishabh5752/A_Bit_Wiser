def cancer_category(percentage):
    if percentage>=0 and percentage<25:
        return "Pre Benign"
    elif percentage>=25 and percentage<50:
        return "Benign"
    elif percentage>=50 and percentage<75:
        return "Pre Malignant"
    else:
        return "Malignant"