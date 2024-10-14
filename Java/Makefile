##
# source directory
##
SRC_DIR := src

##
# output directory
##
OUT_DIR := out

##
# sources
##
SRCS := $(wildcard $(SRC_DIR)/*.java)

##
# classes
##
CLS := $(SRCS:$(SRC_DIR)/%.java=$(OUT_DIR)/%.class)

##
# compiler and compiler flags
##
JC := javac
JCFLAGS := -d $(OUT_DIR)/ -cp $(SRC_DIR)/
J := java

##
# suffixes
##
.SUFFIXES: .java

##
# targets that do not produce output files
##
.PHONY: all clean

##
# default target(s)
##
all: run

$(CLS): $(OUT_DIR)/%.class: $(SRC_DIR)/%.java
	$(JC) $(JCFLAGS) $<

##
# jar
##
jar: $(CLS)
	jar cfe Main.jar Main -C out .

##
# run
##
run: jar
	$(J) -jar Main.jar

##
# clean up any output files
##
clean:
	rm -f $(OUT_DIR)/*.class
	rm -f Main.jar
