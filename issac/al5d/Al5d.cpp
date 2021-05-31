#include "Al5d.hpp"
#include "messages/camera.hpp"

// Linux headers
#include <fcntl.h> // Contains file controls like O_RDWR
#include <errno.h> // Error integer and strerror() function
#include <termios.h> // Contains POSIX terminal control definitions
#include <unistd.h> // write(), read(), close()


namespace isaac {

unsigned char msg[] = {'w'};
int bytes_written = 0;
int serial_port = open("/dev/ttyACM0", O_RDWR);

struct termios tty;

unsigned int microsecond = 1000000;

int i;
int n;
char read_buf[256];

void Al5d::start() {
  tickPeriodically();
  
  // Read in existing settings, and handle any error
  if (tcgetattr(serial_port, &tty) != 0) {
    printf("Error %i from tcgetattr: %s\n", errno, strerror(errno));
  }

  cfsetispeed(&tty, B9600);
  cfsetospeed(&tty, B9600);

  if (tcsetattr(serial_port, TCSANOW, &tty) != 0) {
    printf("Error %i from tcsetattr: %s\n", errno, strerror(errno));
  }

  // Check for errors
  if (serial_port < 0) {
    printf("Error %i from open: %s\n", errno, strerror(errno));
  }

  //for (i = 1; i < 20; ++i) {
  //  bytes_written = write(serial_port, msg, sizeof(msg));
  //  usleep(1 * microsecond);  //sleeps for 3 second
  //}
}

void Al5d::stop() {}

void Al5d::tick() {
  n = read(serial_port, &read_buf, sizeof(read_buf));
  printf("My Name is %s", read_buf);

  LOG_INFO(get_message().c_str());
}

}  // namespace isaac