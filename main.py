import findText


for i in range(1,10,1):
    image_path = "test_v2/test/TEST_000" + str(i) + ".jpg"
    print("===" + image_path + "===")
    findText.find_text(image_path)
