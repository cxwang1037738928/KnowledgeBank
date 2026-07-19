/** db.js — shared PrismaClient singleton (one connection pool per process). */

import { PrismaClient } from '@prisma/client';

export const prisma = new PrismaClient();
