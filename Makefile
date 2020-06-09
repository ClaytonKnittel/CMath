include common.mk

SDIR=src
ODIR=.obj

SLIB=$(LIB_DIR)/libcmath.a


SRC=$(shell find $(SDIR) -type f -name '*.c')
OBJ=$(patsubst $(SDIR)/%.c,$(ODIR)/%.o,$(SRC))
DEP=$(wildcard $(SDIR)/*.h)

DIRS=$(shell find $(SDIR) -type d)
OBJDIRS=$(patsubst $(SDIR)/%,$(ODIR)/%,$(DIRS))

$(shell mkdir -p $(LIB_DIR))
$(shell mkdir -p $(ODIR))
$(shell mkdir -p $(OBJDIRS))
$(shell mkdir -p $(BIN_DIR))

DEPFILES=$(SRC:$(SDIR)/%.c=$(ODIR)/%.d)


.PHONY: all
all: $(SLIB) tests

.PHONY: tests
tests:
	(make -C $(TEST_DIR) BASE_DIR=$(BASE_DIR) SLIB=$(SLIB) LIBCMATH=$(SLIB))


$(SLIB): $(OBJ)
	$(AR) -rcs $@ $^


$(ODIR)/%.o: $(SDIR)/%.c
	$(CC) $(CFLAGS) $< -c -o $@ $(IFLAGS)


-include $(wildcard $(DEPFILES))

.PHONY: clean
clean:
	rm -rf $(ODIR)
	rm -rf $(LIB_DIR)
	rm -rf $(BIN_DIR)
	(make -C $(TEST_DIR) clean)


