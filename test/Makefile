include ../common.mk

SDIR=${CURDIR}
ODIR=${CURDIR}/.obj
PODIR=${CURDIR}/.pobj

SRC=$(shell find $(SDIR) -type f -name '*.c')
OBJ=$(patsubst $(SDIR)/%.c,$(ODIR)/%.o,$(SRC))

PSRC=$(shell find $(SDIR) -type f -name '*.cpp')
POBJ=$(patsubst $(SDIR)/%.cpp,$(PODIR)/%.o,$(PSRC))

CEXE=$(patsubst $(SDIR)/%.c,$(BIN_DIR)/%,$(SRC))
EXE=$(CEXE) $(patsubst $(SDIR)/%.cpp,$(BIN_DIR)/%,$(PSRC))

$(shell mkdir -p $(ODIR))
$(shell mkdir -p $(PODIR))

DEPFILES=$(SRC:$(SDIR)/%.c=$(ODIR)/%.d)
PDEPFILES=$(PSRC:$(SDIR)/%.cpp=$(PODIR)/%.d)


# to prevent files from being auto-deleted by make after build completes
.SECONDARY:

.PHONY: all
all: $(EXE)

$(BIN_DIR)/%: $(ODIR)/%.o $(LIBCMATH)
	$(CC) $(CFLAGS) $^ -o $@ $(IFLAGS) $(LDFLAGS) -lutil

$(ODIR)/%.o: $(SDIR)/%.c
	$(CC) $(CFLAGS) $< -c -o $@ $(IFLAGS)

$(BIN_DIR)/%: $(PODIR)/%.o $(LIBCMATH)
	$(CPPC) $(CPPFLAGS) $^ -o $@ $(IFLAGS) $(LDFLAGS) -lutil

$(PODIR)/%.o: $(SDIR)/%.cpp
	$(CPPC) $(CPPFLAGS) $< -c -o $@ $(IFLAGS)


-include $(wildcard $(DEPFILES))
-include $(wildcard $(PDEPFILES))

.PHONY: clean
clean:
	rm -rf $(ODIR)
	rm -rf $(PODIR)

