include common.mk

SDIR=src
IDIR=include
ODIR=.obj

SLIB=$(LIB_DIR)/libcmath.a


# C src
SRC=$(shell find $(SDIR) -type f -name '*.c')
OBJ=$(patsubst $(SDIR)/%.c,$(ODIR)/%.o,$(SRC))

# C++ src
PSRC=$(shell find $(SDIR) -type f -name '*.cpp')
POBJ=$(patsubst $(SDIR)/%.cpp,$(ODIR)/%.o,$(PSRC))

DEP=$(wildcard $(IDIR)/*.h)

DIRS=$(shell find $(SDIR) -type d)
OBJDIRS=$(patsubst $(SDIR)/%,$(ODIR)/%,$(DIRS))

$(shell mkdir -p $(LIB_DIR))
$(shell mkdir -p $(ODIR))
$(shell mkdir -p $(OBJDIRS))
$(shell mkdir -p $(BIN_DIR))

DEPFILES=$(SRC:$(SDIR)/%.c=$(ODIR)/%.d)
PDEPFILES=$(PSRC:$(SDIR)/%.cpp=$(ODIR)/%.d)


.PHONY: all
all: $(SLIB) tests

.PHONY: tests
tests:
	(make -C $(TEST_DIR) BASE_DIR=$(BASE_DIR) SLIB=$(SLIB) LIBCMATH=$(SLIB))


$(SLIB): $(OBJ) $(POBJ)
	$(AR) -rcs $@ $^


$(ODIR)/%.o: $(SDIR)/%.c
	$(CC) $(CFLAGS) $< -c -o $@ $(IFLAGS)

$(ODIR)/%.o: $(SDIR)/%.cpp
	$(CPPC) $(CPPFLAGS) $< -c -o $@ $(IFLAGS)


-include $(wildcard $(DEPFILES))
-include $(wildcard $(PDEPFILES))

.PHONY: clean
clean:
	rm -rf $(ODIR)
	rm -rf $(LIB_DIR)
	rm -rf $(BIN_DIR)
	(make -C $(TEST_DIR) clean)


