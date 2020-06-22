
.obj/pmath/vec.o:	file format Mach-O 64-bit x86-64

Disassembly of section __TEXT,__text:
__vec_add:
       0:	55 	pushq	%rbp
       1:	48 89 e5 	movq	%rsp, %rbp
       4:	5d 	popq	%rbp
       5:	c3 	retq
       6:	66 2e 0f 1f 84 00 00 00 00 00 	nopw	%cs:(%rax,%rax)

__fvec_add:
      10:	55 	pushq	%rbp
      11:	48 89 e5 	movq	%rsp, %rbp
      14:	48 83 f9 40 	cmpq	$64, %rcx
      18:	0f 87 0d 01 00 00 	ja	269 <__fvec_add+0x11b>
      1e:	41 89 c8 	movl	%ecx, %r8d
      21:	41 80 e0 07 	andb	$7, %r8b
      25:	48 85 c9 	testq	%rcx, %rcx
      28:	0f 84 ec 00 00 00 	je	236 <__fvec_add+0x10a>
      2e:	31 c0 	xorl	%eax, %eax
      30:	c5 fc 10 04 86 	vmovups	(%rsi,%rax,4), %ymm0
      35:	c5 fc 58 04 82 	vaddps	(%rdx,%rax,4), %ymm0, %ymm0
      3a:	c5 fc 11 04 87 	vmovups	%ymm0, (%rdi,%rax,4)
      3f:	8d 40 08 	leal	8(%rax), %eax
      42:	48 39 c8 	cmpq	%rcx, %rax
      45:	72 e9 	jb	-23 <__fvec_add+0x20>
      47:	41 89 c1 	movl	%eax, %r9d
      4a:	45 84 c0 	testb	%r8b, %r8b
      4d:	0f 84 d3 00 00 00 	je	211 <__fvec_add+0x116>
      53:	41 0f b6 c0 	movzbl	%r8b, %eax
      57:	48 ff c8 	decq	%rax
      5a:	48 83 f8 07 	cmpq	$7, %rax
      5e:	0f 83 cc 00 00 00 	jae	204 <__fvec_add+0x120>
      64:	80 e1 07 	andb	$7, %cl
      67:	fe c9 	decb	%cl
      69:	80 f9 06 	cmpb	$6, %cl
      6c:	0f 87 b4 00 00 00 	ja	180 <__fvec_add+0x116>
      72:	0f b6 c1 	movzbl	%cl, %eax
      75:	48 8d 0d f4 00 00 00 	leaq	244(%rip), %rcx
      7c:	48 63 04 81 	movslq	(%rcx,%rax,4), %rax
      80:	48 01 c8 	addq	%rcx, %rax
      83:	ff e0 	jmpq	*%rax
      85:	c4 a1 7a 10 44 8e 18 	vmovss	24(%rsi,%r9,4), %xmm0
      8c:	c4 a1 7a 58 44 8a 18 	vaddss	24(%rdx,%r9,4), %xmm0, %xmm0
      93:	c4 a1 7a 11 44 8f 18 	vmovss	%xmm0, 24(%rdi,%r9,4)
      9a:	c4 a1 7a 10 44 8e 14 	vmovss	20(%rsi,%r9,4), %xmm0
      a1:	c4 a1 7a 58 44 8a 14 	vaddss	20(%rdx,%r9,4), %xmm0, %xmm0
      a8:	c4 a1 7a 11 44 8f 14 	vmovss	%xmm0, 20(%rdi,%r9,4)
      af:	c4 a1 7a 10 44 8e 10 	vmovss	16(%rsi,%r9,4), %xmm0
      b6:	c4 a1 7a 58 44 8a 10 	vaddss	16(%rdx,%r9,4), %xmm0, %xmm0
      bd:	c4 a1 7a 11 44 8f 10 	vmovss	%xmm0, 16(%rdi,%r9,4)
      c4:	c4 a1 7a 10 44 8e 0c 	vmovss	12(%rsi,%r9,4), %xmm0
      cb:	c4 a1 7a 58 44 8a 0c 	vaddss	12(%rdx,%r9,4), %xmm0, %xmm0
      d2:	c4 a1 7a 11 44 8f 0c 	vmovss	%xmm0, 12(%rdi,%r9,4)
      d9:	c4 a1 7a 10 44 8e 08 	vmovss	8(%rsi,%r9,4), %xmm0
      e0:	c4 a1 7a 58 44 8a 08 	vaddss	8(%rdx,%r9,4), %xmm0, %xmm0
      e7:	c4 a1 7a 11 44 8f 08 	vmovss	%xmm0, 8(%rdi,%r9,4)
      ee:	c4 a1 7a 10 44 8e 04 	vmovss	4(%rsi,%r9,4), %xmm0
      f5:	c4 a1 7a 58 44 8a 04 	vaddss	4(%rdx,%r9,4), %xmm0, %xmm0
      fc:	c4 a1 7a 11 44 8f 04 	vmovss	%xmm0, 4(%rdi,%r9,4)
     103:	c4 a1 7a 10 04 8e 	vmovss	(%rsi,%r9,4), %xmm0
     109:	c4 a1 7a 58 04 8a 	vaddss	(%rdx,%r9,4), %xmm0, %xmm0
     10f:	c4 a1 7a 11 04 8f 	vmovss	%xmm0, (%rdi,%r9,4)
     115:	5d 	popq	%rbp
     116:	c5 f8 77 	vzeroupper
     119:	c3 	retq
     11a:	45 31 c9 	xorl	%r9d, %r9d
     11d:	45 84 c0 	testb	%r8b, %r8b
     120:	0f 85 2d ff ff ff 	jne	-211 <__fvec_add+0x43>
     126:	5d 	popq	%rbp
     127:	c5 f8 77 	vzeroupper
     12a:	c3 	retq
     12b:	e8 00 00 00 00 	callq	0 <__fvec_add+0x120>
     130:	48 8b 05 00 00 00 00 	movq	(%rip), %rax
     137:	48 8b 38 	movq	(%rax), %rdi
     13a:	48 8d 35 5d 00 00 00 	leaq	93(%rip), %rsi
     141:	ba 20 00 00 00 	movl	$32, %edx
     146:	31 c0 	xorl	%eax, %eax
     148:	c5 f8 77 	vzeroupper
     14b:	e8 00 00 00 00 	callq	0 <__fvec_add+0x140>
     150:	48 8d 3d bc 00 00 00 	leaq	188(%rip), %rdi
     157:	48 8d 35 2e 00 00 00 	leaq	46(%rip), %rsi
     15e:	48 8d 0d 37 00 00 00 	leaq	55(%rip), %rcx
     165:	ba 20 00 00 00 	movl	$32, %edx
     16a:	e8 00 00 00 00 	callq	0 <__fvec_add+0x15f>
     16f:	90 	nop
     170:	93 	xchgl	%ebx, %eax
     171:	ff ff  <unknown>
     173:	ff 7e ff  <unknown>
     176:	ff ff  <unknown>
     178:	69 ff ff ff 54 ff 	imull	$4283760639, %edi, %edi
     17e:	ff ff  <unknown>
     180:	3f  <unknown>
     181:	ff ff  <unknown>
     183:	ff 2a 	ljmpl	*(%rdx)
     185:	ff ff  <unknown>
     187:	ff 15  <unknown>
     189:	ff ff  <unknown>
     18b:	ff  <unknown>
