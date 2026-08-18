#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define CF_BREAK		0
#define CF_CONTINUE		1

#define CS_INITIAL		0
#define CS_SYNCED		1

int search_for_packet(int* state, FILE* in)
{
	uint8_t buf = 0;
	if (fread(&buf, 1, 1, in) != 1) return CF_BREAK;

	if (buf == 0xff)
	{
		fseek(in, -1, SEEK_CUR);
		*state = CS_SYNCED;
	}

	return CF_CONTINUE;
}

int write_packet(uint8_t* frame, FILE* out, int i)
{
	if (i & 1)
	{
		frame[2] = 0xc0;
		if (fwrite(frame, 416, 1, out) != 1) return CF_BREAK;
	}

	else
	{
		frame[2] = 0xc2;
		if (fwrite(frame, 420, 1, out) != 1) return CF_BREAK;
	}

	return CF_CONTINUE;
}

int core_mpp_to_mp1(int* state, FILE* in, FILE* out, int i)
{
	if (*state == CS_INITIAL)
	{
		return search_for_packet(state, in);
	}

	else if (*state == CS_SYNCED)
	{
		uint8_t buf[420];
		if (fread(&buf, 420, 1, in) != 1) return CF_BREAK;

		if (buf[0] != 0xff)
		{
			fprintf(stderr, __FUNCTION__ ": warning: out of sync\n");

			fseek(in, -419, SEEK_CUR);
			*state = CS_INITIAL;
			return CF_CONTINUE;
		}

		if (buf[2] == 0xc0 || buf[2] == 0xc2)
		{
			if (buf[2] == 0xc2)
				fprintf(stderr, __FUNCTION__ ": warning: padding bit in a packet\n");

			return write_packet(buf, out, i);
		}

		else
			fprintf(stderr, __FUNCTION__ ": warning: skipping invalid packet\n");
	}

	return CF_CONTINUE;
}

int core_mp1_fix(int* state, FILE* in, FILE* out, int i)
{
	if (*state == CS_INITIAL)
	{
		return search_for_packet(state, in);
	}

	else if (*state == CS_SYNCED)
	{
		uint8_t buf[416];
		if (fread(&buf, 416, 1, in) != 1) return CF_BREAK;

		if (buf[0] != 0xff)
		{
			fprintf(stderr, __FUNCTION__ ": warning: out of sync\n");

			fseek(in, -415, SEEK_CUR);
			*state = CS_INITIAL;
			return CF_CONTINUE;
		}

		if (buf[2] == 0xc0 || buf[2] == 0xc2)
		{
			if (buf[2] == 0xc2)
			{
				if (fseek(in, 4, SEEK_CUR)) return CF_BREAK;
			}

			return write_packet(buf, out, i);
		}

		else
			fprintf(stderr, __FUNCTION__ ": warning: skipping invalid packet\n");
	}

	return CF_CONTINUE;
}

int process_file(
	const char* name_in,
	const char* name_out,
	int (*proc_core)(int*, FILE*, FILE*, int))
{
	FILE* f_in = fopen(name_in, "rb");

	if (!f_in)
		return -1;

	FILE* f_out = fopen(name_out, "wb");
	if (!f_out)
	{
		fclose(f_in);
		return -1;
	}

	int i = 0;
	int packets = 0;
	int state = CS_INITIAL;

	while (proc_core(&state, f_in, f_out, i) == CF_CONTINUE)
	{
		i++;

		if(state == CS_SYNCED)
			packets++;

		if (packets > 49)
		{
			packets = 1;
			i++;
		}
	}

	fclose(f_in);
	fclose(f_out);

	return 0;
}

int handle_name(const char* name)
{
	size_t nl = strlen(name);
	char* ext = (char*)(((intptr_t)name) + nl - 4);

	if (nl < 4 || ext[0] != '.')
		return -1;

	char tmp[0x200];
	strcpy(tmp, name);

	if (!strcmp(".MPP", ext) || !strcmp(".mpp", ext))
	{
		tmp[nl - 1] = '1';
		tmp[nl - 2] = 'p';
		tmp[nl - 3] = 'm';

		process_file(name, tmp, core_mpp_to_mp1);
	}

	else if (!strcmp(".mp1", ext) || !strcmp(".MP1", ext))
	{
		tmp[nl - 4] = '\0';
		char tmp2[0x210];
		sprintf(tmp2, "%s-new.mp1", tmp);

		process_file(name, tmp2, core_mp1_fix);
	}

	else
		return -1;

	return 0;
}

int main(int c, char** v)
{
	fprintf(stderr, "Quick and shitty alternative to DCCU tool\n"
		"made to produce bitrate-accurate mp1 files (and to fix existing ones).\n"
		"No support for MP1->MPP back conversion.\n"
		"Do not feed with 32 or 48 kHz files (they're fine and don't need this kind of processing).\n"
		"[Version 1.1]\n");

	for (int i = 1; i < c; i++)
	{
		if (handle_name(v[i]))
			fprintf(stderr, "! Something gone wrong with %s.\n", v[i]);
		else
			fprintf(stderr, "> Processed %s\n", v[i]);
	}

	return 0;
}
