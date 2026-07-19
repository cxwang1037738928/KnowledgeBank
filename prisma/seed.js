/**
 * seed.js — creates the admin account (admin@gmail.com / admin123).
 * Run: npm run db:seed (idempotent — upserts by email).
 */

import { PrismaClient } from '@prisma/client';
import bcrypt from 'bcryptjs';

const prisma = new PrismaClient();

async function main() {
  const password = await bcrypt.hash('admin123', 10);
  const admin = await prisma.user.upsert({
    where: { email: 'admin@gmail.com' },
    update: { isAdmin: true },
    create: { email: 'admin@gmail.com', password, isAdmin: true },
  });
  console.log(`[seed] admin user ready (id=${admin.id}, email=${admin.email})`);
}

main()
  .catch((err) => { console.error('[seed]', err); process.exit(1); })
  .finally(() => prisma.$disconnect());
