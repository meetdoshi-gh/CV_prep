import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import time

###############################################################
# Load Model
###############################################################
device = "cuda" if torch.cuda.is_available() else "cpu"

weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
model = model.to(device)
model.eval()

labels = weights.meta["categories"]


###############################################################
# Preprocessing
###############################################################
preprocessing = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


###############################################################
# Inference Loop
###############################################################
cap = cv2.VideoCapture(0)
start_time = 0

while 1:
    ret, frame = cap.read()
    if not ret:
        break


    # cv2 based preprocess
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # rgb = cv2.bilateralFilter(rgb, 3, 8, 8)

    # preprocess
    input_tensor = preprocessing(rgb).unsqueeze(0).to(device)

    # infrence 
    with torch.no_grad():
        logits = model(input_tensor)
    probs = torch.nn.functional.softmax(logits[0], dim=0)

    # Highest
    k=1
    confidence, idx = torch.max(probs, dim=0)
    category = labels[idx]
    

    # # Top K
    # k=3
    # confidence, idx = torch.topk(probs, k=k, dim=0)
    # category = [labels[x] for x in idx]

    # FPS
    curr_time = time.time()
    fps = 1/(curr_time - start_time)
    start_time = curr_time

    # Show Output
    text = ""
    if k==1:
        text = f"{category}: {confidence:.2f}"
        cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0,255,0),
        2)
    else:   
        for i in range(k):
            text = f"{category[i]}: {confidence[i]:.2f}"
            cv2.putText(
            frame,
            text,
            (20, 40+(i*15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )

    text = f"FPS: {fps:0.1f}"
    cv2.putText(
        frame,
        text,
        (520, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,0,0),
        2
    )

    cv2.imshow("Classification Webcam", frame)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
