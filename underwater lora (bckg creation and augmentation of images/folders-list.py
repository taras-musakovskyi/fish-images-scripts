import zipfile

with zipfile.ZipFile("r_caustic_dataset.zip") as z:
    for name in z.namelist()[:50]:
        print(name)

