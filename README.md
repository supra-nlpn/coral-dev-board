#Coral Dev Board

## Coral Dev Board Setup Guide (v1.0 - Model 2019)

## 1. Collect Required Hardware

- **Coral Dev Board**: Version 1.0, model-2019  
- **Host Computer**: Running Linux (recommended), with Python 3 and pip installed  
  - _Example_: Dell Precision 5820 running Ubuntu 24.04.2  
- **microSD Card**: Minimum 8GB capacity (e.g., Sandisk 32 GB) and adapter  
- **USB-C to USB-A Cable**: For data transfer between the board and host computer  
- **USB-C Power Supply**: 3A/5V (e.g., a standard phone charger)  
- **Network Connection**: Ethernet cable or Wi-Fi for board connectivity  
- _Note_: Mouse and keyboard are not required for this setup  

---

## 2. Flash the Board

### On Host Machine

1. **Download and Unzip the SD Card Image**:  
   - [enterprise-eagle-flashcard-20211117215217.zip](https://dl.google.com/coral/mendel/enterprise/enterprise-eagle-flashcard-20211117215217.zip)  
   - Unzip to obtain `flashcard_arm64.img`

2. **Download and Unzip Balena Etcher**:  
   - [balenaEtcher-linux-x64-2.1.2.zip](https://github.com/balena-io/etcher/releases/download/v2.1.2/balenaEtcher-linux-x64-2.1.2.zip)  
   - Extract and navigate to the extracted folder  
   - Locate and double-click the `balenaEtcher` executable to launch the application  

3. **Flash the SD Card**:  
   - In Balena Etcher:  
     - Select Image: `flashcard_arm64.img`  
     - Select Drive: Your inserted microSD card  
     - Click **Flash!**  
   - This process typically takes 5–10 minutes  

### On Dev Board

1. **Set Boot Mode to SD Card**:  
   - Configure the boot mode switches as follows:  
     - Switch 1: ON  
     - Switch 2: OFF  
     - Switch 3: ON  
     - Switch 4: ON  

2. **Insert the Flashed microSD Card**:  
   - Ensure the card's pins face toward the board  

3. **Power the Board**:  
   - Connect the USB-C power supply to the port labeled `PWR`  
   - The red LED should illuminate, indicating the board is flashing the eMMC from the SD card  
   - Wait approximately 5–10 minutes for the process to complete  

4. **Wait for Flashing to Complete**:  
   - The red LED will turn off once flashing is complete  

5. **Finalize Setup**:  
   - Unplug the power supply  
   - Remove the microSD card  
   - Reconfigure the boot mode switches to boot from eMMC:  
     - Switch 1: ON  
     - Switch 2: OFF  
     - Switch 3: OFF  
     - Switch 4: OFF  

6. **Reboot the Board**:  
   - Reconnect the power supply  
   - The board should now boot into Mendel Linux  
   - The initial boot may take approximately 3 minutes  

---

## 3. Install MDT (Mendel Development Tool)

### On Host Machine

1. **Create a Virtual Environment**:

   ```bash
   python3 -m venv ~/mdt-env
   ```

2. **Activate the Virtual Environment**:

   ```bash
   source ~/mdt-env/bin/activate
   ```

3. **Install MDT**:

   ```bash
   pip install mendel-development-tool
   ```

4. **Verify Installation**:

   ```bash
   mdt devices
   mdt shell
   ```

---

## 4. Connect to the Board's Shell via MDT

1. **Establish USB Connection**:  
   - Connect a USB-C cable from your host computer to the board's `OTG` port  

2. **Detect the Device**:

   ```bash
   mdt devices
   ```

   - Expected output:

     ```
     hopeful-apple        (192.168.100.2)
     ```

   - Note: The hostname is randomly generated upon first boot  

3. **Access the Board's Shell**:

   ```bash
   mdt shell
   ```

   - You should see the prompt:

     ```
     mendel@hopeful-apple:~$
     ```

---

## 5. Connect the Board to the Internet

1. **Establish Network Connection**:  
   - Connect an Ethernet cable to the board  
   - For Wi-Fi setup, refer to the [official documentation](https://coral.ai/docs/dev-board/get-started/#connect-to-internet)  

2. **Verify Network Connection**:

   ```bash
   nmcli connection show
   ```

   - Sample output:

     ```
     NAME           UUID                                  TYPE       DEVICE
     NetworkName    61f5d6b2-5f52-4256-83ae-7f148546575a   ethernet   eth0
     ```

---

## 6. Update the Mendel Software

1. **Update Package Lists**:

   ```bash
   sudo apt-get update
   ```

2. **Upgrade Installed Packages**:

   ```bash
   sudo apt-get dist-upgrade
   ```

---

## 7. Run a Model Using the PyCoral API

1. **Download Example Code**:

   ```bash
   mkdir coral && cd coral
   git clone https://github.com/google-coral/pycoral.git
   cd pycoral
   ```

2. **Install Dependencies and Download Model/Data**:

   ```bash
   bash examples/install_requirements.sh classify_image.py
   ```

3. **Run the Image Classifier**:

   ```bash
   python3 examples/classify_image.py      --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite      --labels test_data/inat_bird_labels.txt      --input test_data/parrot.jpg
   ```

4. **Expected Output**:

   ```
   ----INFERENCE TIME----
   Note: The first inference on Edge TPU is slow because it includes loading the model into Edge TPU memory.
   13.1ms
   2.7ms
   3.1ms
   3.2ms
   3.1ms
   -------RESULTS--------
   Ara macao (Scarlet Macaw): 0.75781
   ```

---

You have now successfully set up the Coral Dev Board and performed an inference using the PyCoral API. For further exploration, try deploying your own models, experimenting with additional examples, or training and compiling custom models with the Edge TPU Compiler.
