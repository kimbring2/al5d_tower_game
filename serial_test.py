import time
import serial

# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial('/dev/ttyUSB0')

ser.isOpen()

print('Enter your commands below.\r\nInsert "exit" to leave the application.')

#input=1
while 1 :
    # get keyboard input
    #input = raw_input(">> ")
    # Python 3 users
    input_command = input(">> ")
    input_command = input_command.encode()
    print("input_command: " + str(input_command))
    if input == 'exit':
        ser.close()
        exit()
    else:
        # send the character to the device
        # (note that I happend a \r\n carriage return and line feed to the characters - this is requested by my device)
        ser.write(input_command)
        #ser.write(b'w')
        #out = ''
        # let's wait one second before reading output (let's give device time to answer)
        #time.sleep(0.5)
        #while ser.inWaiting() > 0:
        #    out += ser.read(1)

        #if out != '':
        #    print(">>" + out)